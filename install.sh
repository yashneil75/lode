#!/bin/sh
# Install a lode standalone binary.
#
#   curl -fsSL https://raw.githubusercontent.com/yashneil75/lode/main/install.sh | sh
#
# Everything can be overridden with environment variables for air-gapped or custom
# hosting setups:
#
#   LODE_VERSION      release tag to install, e.g. v0.1.0 (default: latest)
#   LODE_REPO         owner/name on GitHub (default: yashneil75/lode)
#   LODE_BASE_URL     where the archives live (default: the release download host)
#   LODE_INSTALL_DIR  where the app folder goes (default: ~/.local/share/lode)
#   LODE_BIN_DIR      where the `lode` launcher goes (default: ~/.local/bin)

set -eu

REPO="${LODE_REPO:-yashneil75/lode}"
VERSION="${LODE_VERSION:-}"
INSTALL_DIR="${LODE_INSTALL_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/lode}"
BIN_DIR="${LODE_BIN_DIR:-${XDG_BIN_HOME:-$HOME/.local/bin}}"

say() { printf '  %s\n' "$*"; }
die() { printf 'lode install: %s\n' "$*" >&2; exit 1; }

need() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}
need curl
need tar

# --- platform -------------------------------------------------------------
os="$(uname -s 2>/dev/null || echo unknown)"
case "$os" in
    Linux) target_os=linux ;;
    Darwin) target_os=darwin ;;
    *) die "no standalone build for $os. Install with: pipx install lode-code-search" ;;
esac

arch="$(uname -m 2>/dev/null || echo unknown)"
case "$arch" in
    x86_64 | amd64) target_arch=x86_64 ;;
    arm64 | aarch64) target_arch=arm64 ;;
    *) die "no standalone build for $arch. Install with: pipx install lode-code-search" ;;
esac

# Apple Silicon runs x86_64 binaries under Rosetta, but we ship the arm64 build, so
# a translated shell must not be allowed to pick the wrong asset.
if [ "$target_os" = darwin ] && [ "$target_arch" = x86_64 ]; then
    if [ "$(sysctl -n sysctl.proc_translated 2>/dev/null || echo 0)" = "1" ]; then
        target_arch=arm64
    fi
fi

if [ "$target_os" = darwin ] && [ "$target_arch" = x86_64 ]; then
    die "no Intel macOS build. Install with: pipx install lode-code-search"
fi

# --- resolve the version --------------------------------------------------
if [ -z "$VERSION" ]; then
    VERSION="$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" \
        | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -1)"
    [ -n "$VERSION" ] || die "could not determine the latest release of $REPO; set LODE_VERSION"
fi
say "target ${target_os}-${target_arch} ${VERSION}"

BASE_URL="${LODE_BASE_URL:-https://github.com/${REPO}/releases/download/${VERSION}}"
ASSET="lode-${VERSION#v}-${target_os}-${target_arch}.tar.gz"
URL="${BASE_URL}/${ASSET}"

# --- download and verify --------------------------------------------------
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
say "downloading ${ASSET}"
curl -fsSL --retry 3 -o "${tmp}/${ASSET}" "$URL" || die "download failed: $URL"
curl -fsSL -o "${tmp}/sha256sums.txt" "${BASE_URL}/sha256sums.txt" \
    || say "no sha256sums.txt at ${BASE_URL}; skipping verification"

if [ -f "${tmp}/sha256sums.txt" ]; then
    if command -v sha256sum >/dev/null 2>&1; then
        checker="sha256sum"
    elif command -v shasum >/dev/null 2>&1; then
        checker="shasum -a 256"
    else
        die "no sha256 tool available to verify the download"
    fi
    expected="$(awk -v asset="$ASSET" '$2 == asset {print $1; exit}' "${tmp}/sha256sums.txt")"
    [ -n "$expected" ] || die "${ASSET} is not listed in sha256sums.txt"
    actual="$(cd "$tmp" && $checker "$ASSET" | awk '{print $1}')"
    [ "$actual" = "$expected" ] || die "checksum mismatch: $actual != $expected"
    say "checksum ok"
fi

# --- install --------------------------------------------------------------
# The archive holds a `lode/` folder; replace the old one only after the new one is
# unpacked, so a failed download cannot leave you without a binary.
say "installing to ${INSTALL_DIR}"
mkdir -p "$tmp/stage"
tar -xzf "${tmp}/${ASSET}" -C "$tmp/stage"
[ -x "${tmp}/stage/lode/lode" ] || die "archive did not contain lode/lode"

mkdir -p "$(dirname "$INSTALL_DIR")"
# Written as ifs, not `[ -d x ] && ...`: under `set -e` a false test at the end of an
# && list is a failed command, and the script would exit on the happy path.
if [ -d "$INSTALL_DIR" ]; then
    rm -rf "${INSTALL_DIR}.old"
    mv "$INSTALL_DIR" "${INSTALL_DIR}.old"
    if ! mv "${tmp}/stage/lode" "$INSTALL_DIR"; then
        mv "${INSTALL_DIR}.old" "$INSTALL_DIR"
        die "could not move the new build into place"
    fi
    rm -rf "${INSTALL_DIR}.old"
else
    mv "${tmp}/stage/lode" "$INSTALL_DIR"
fi

mkdir -p "$BIN_DIR"
ln -sf "${INSTALL_DIR}/lode" "${BIN_DIR}/lode"
say "linked ${BIN_DIR}/lode"

# --- gatekeeper -----------------------------------------------------------
if [ "$target_os" = darwin ]; then
    # Quarantine survives the download; without this the first run is blocked.
    if ! xattr -dr com.apple.quarantine "$INSTALL_DIR" 2>/dev/null; then
        say "if macOS refuses to run it: xattr -dr com.apple.quarantine ${INSTALL_DIR}"
    fi
fi

if ! command -v lode >/dev/null 2>&1; then
    say ""
    say "add ${BIN_DIR} to your PATH, then reopen the terminal:"
    say "  export PATH=\"${BIN_DIR}:\$PATH\""
fi

say ""
"${BIN_DIR}/lode" version || die "the installed binary did not run"
say "next: lode index ~/code/your-project -n your-project"
