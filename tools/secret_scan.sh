#!/bin/sh
# Second secret scan, next to tools/privacy_scan.py: wraps Betterleaks.
#
#   sh tools/secret_scan.sh staged         the index, as the next commit would contain it
#   sh tools/secret_scan.sh message FILE   one commit message file
#   sh tools/secret_scan.sh history        every commit diff on any ref, plus all commit
#                                          and tag messages (Betterleaks itself scans diffs only)
#
# Exit status: 0 clean, 2 usage error, 3 betterleaks missing, anything else: findings or a
# betterleaks failure. Secret values are never printed (--redact); a finding is one line,
# "rule path:line commit". Validation is off (it would send candidates to the network) and the
# repository's own Betterleaks config is not loaded (--unsafe-auto-load-config is not used).
set -eu

usage() {
  echo "usage: sh tools/secret_scan.sh staged | message FILE | history" >&2
  exit 2
}

[ "$#" -ge 1 ] || usage
mode=$1
case "$mode" in
  staged | history) [ "$#" -eq 1 ] || usage ;;
  message) [ "$#" -eq 2 ] && [ -r "$2" ] || usage ;;
  *) usage ;;
esac

if ! command -v betterleaks >/dev/null 2>&1; then
  echo "secret_scan: betterleaks is not on PATH, so the secret scan cannot run (a guard that skips itself is no guard)." >&2
  echo "secret_scan: install it with 'brew install betterleaks', or from https://github.com/betterleaks/betterleaks/releases" >&2
  exit 3
fi

here=$(cd "$(dirname "$0")" && pwd)
template="$here/secret_scan.tmpl"

# scan LABEL SUBCOMMAND [ARGS...]: stdin, when the subcommand reads it, is passed through.
scan() {
  label=$1
  shift
  rc=0
  out=$(betterleaks "$@" --redact --no-banner --no-color --log-level warn \
    --report-format template --report-template "$template" --report-path -) || rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "secret_scan: $label: betterleaks exited $rc" >&2
    if [ -n "$out" ]; then
      printf '%s\n' "$out" | sed 's/^/  /' >&2
    fi
    echo "secret_scan: values are redacted. For a false positive see 'Second scan: Betterleaks' in PRIVACY.md." >&2
    return 1
  fi
}

case "$mode" in
  staged)
    scan "staged changes" git --pre-commit --staged .
    ;;
  message)
    scan "commit message" stdin <"$2"
    ;;
  history)
    scan "history" git .
    if [ -n "$(git for-each-ref --count=1)" ]; then
      commits=$(git log --all --format='%B')
      tags=$(git for-each-ref --format='%(contents)' refs/tags)
      printf '%s\n%s\n' "$commits" "$tags" | scan "commit and tag messages" stdin
    fi
    ;;
esac
