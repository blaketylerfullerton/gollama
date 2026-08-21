#!/bin/sh
# Installs the latest gollama release for this OS/arch and (if run from a
# terminal) launches it. Usage:
#   curl -fsSL https://raw.githubusercontent.com/blaketylerfullerton/gollama/main/install.sh | sh
set -e

REPO="blaketylerfullerton/gollama"
BIN_NAME="gollama"

say() { printf '%s\n' "$*" >&2; }
die() { say "error: $*"; exit 1; }

need() { command -v "$1" >/dev/null 2>&1 || die "'$1' is required but not installed"; }
need curl
need tar

os=$(uname -s)
case "$os" in
  Linux) goos="linux" ;;
  Darwin) goos="darwin" ;;
  *) die "unsupported OS: $os (on Windows, use install.ps1 instead: see the README)" ;;
esac

arch=$(uname -m)
case "$arch" in
  x86_64|amd64) goarch="amd64" ;;
  arm64|aarch64) goarch="arm64" ;;
  *) die "unsupported architecture: $arch" ;;
esac

say "Looking up the latest release of $REPO..."
latest_url="https://api.github.com/repos/$REPO/releases/latest"
tag=$(curl -fsSL "$latest_url" | grep '"tag_name":' | head -n1 | sed -E 's/.*"tag_name": *"([^"]+)".*/\1/')
[ -n "$tag" ] || die "could not determine the latest release (checked $latest_url)"
version=${tag#v}

asset="${BIN_NAME}_${version}_${goos}_${goarch}.tar.gz"
url="https://github.com/$REPO/releases/download/$tag/$asset"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

say "Downloading $asset ($tag)..."
curl -fsSL "$url" -o "$tmp/$asset" || die "failed to download $url"

tar -xzf "$tmp/$asset" -C "$tmp"
[ -f "$tmp/$BIN_NAME" ] || die "archive didn't contain a '$BIN_NAME' binary"
chmod +x "$tmp/$BIN_NAME"

# Prefer /usr/local/bin when writable; fall back to ~/.local/bin.
if [ -w "/usr/local/bin" ]; then
  install_dir="/usr/local/bin"
else
  install_dir="$HOME/.local/bin"
  mkdir -p "$install_dir"
fi

mv "$tmp/$BIN_NAME" "$install_dir/$BIN_NAME"
say "Installed $BIN_NAME $version to $install_dir/$BIN_NAME"

case ":$PATH:" in
  *":$install_dir:"*) ;;
  *) say "Note: $install_dir isn't on your PATH. Add this to your shell profile:"
     say "  export PATH=\"$install_dir:\$PATH\"" ;;
esac

# Drop straight into the app if we're actually in a terminal. Piping this
# script through `sh` consumes stdin, so re-point it at the tty directly;
# if there isn't one (e.g. a non-interactive CI run), just stop here.
if [ -t 1 ] && [ -r /dev/tty ]; then
  exec "$install_dir/$BIN_NAME" < /dev/tty
else
  say "Run '$BIN_NAME' to get started."
fi
