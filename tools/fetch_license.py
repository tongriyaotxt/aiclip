import json
import subprocess
import sys

raw = subprocess.run(["gh", "api", "/licenses/apache-2.0"],
                     capture_output=True, text=True, encoding="utf-8")
if raw.returncode != 0:
    sys.exit(raw.stderr)
data = json.loads(raw.stdout)
body = data["body"]
if "Apache License" not in body:
    sys.exit("unexpected body")
with open("LICENSE", "w", encoding="utf-8", newline="\n") as f:
    f.write(body.rstrip() + "\n")
print("LICENSE written:", len(body), "chars,", body.count("\n"), "lines")
