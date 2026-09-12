"""Local operational handoff for six frozen fits; no scoring or scientific edits.

Uses the existing SSH known_hosts and prompts once for the remote password.
All arrays/checkpoints remain intact. A failed check stops the coordinator.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import getpass
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import socket
import subprocess
import sys
import tarfile
import time
import uuid


HOST, PORT, USER = "connect.westc.seetacloud.com", 55786, "root"
BASE = "/root/physical_identity_mainline_v1"
SOURCE, OUTPUT, EXPORTS, JOBS = (BASE + "/" + n for n in ("source", "results", "exports", "jobs"))
DESIGN = "a86cadbfcafe8abc9a12810c3ee71d673c5a1bae55c018824fb952e07bdac9e4"
PREPARED_COMMIT = "dbf304f86563442c6be6f064e8cd72fabee2e682"
BRANCH = "refs/heads/codex/v58-v59-run-records"
CASES = tuple(f"{kind}__{pool}_R{rep}_K125" for rep in (71, 72)
              for kind, pool in (("SUPPORTED_MISMATCH", "CAL"), ("close_neighbor", "HOLD"),
                                 ("relatively_isolated", "HOLD")))
COMMON = {"design.json", "design_seal.json", "donor_split.json", "target_containment.json", "model_patterns.json"}
TEXT_SUFFIXES = {".json", ".txt", ".csv", ".tsv", ".log", ".md", ".yaml", ".yml"}
REQUIRED_CASE_FILES = {"training_complete.json", "case_review.json", "candidate_records.json", "molecular_records.json"}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def immutable(path, data):
    require(not path.is_symlink(), "LOCAL_SYMLINK:" + str(path))
    if path.exists():
        require(path.is_file() and path.read_bytes() == data, "LOCAL_OVERWRITE_CONFLICT:" + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def safe_member(root, name):
    p = PurePosixPath(name)
    require(name and not p.is_absolute() and p.as_posix() == name and
            all(x not in ("", ".", "..") for x in p.parts) and
            "\\" not in name and ":" not in name and "\x00" not in name, "UNSAFE_ARCHIVE_PATH")
    result = root.joinpath(*p.parts)
    require(result.resolve().is_relative_to(root.resolve()), "LOCAL_PATH_ESCAPE")
    for parent in (result, *result.parents):
        if parent == root.parent:
            break
        require(not parent.is_symlink(), "LOCAL_SYMLINK_PATH")
    return result


class Remote:
    """Retry only read-only operations; launch RPCs implement their own receipt guard."""

    def __init__(self, password):
        import paramiko
        self.module, self.password, self.client = paramiko, password, None
        self.connect()

    def connect(self):
        self.close()
        client = self.module.SSHClient()
        client.load_system_host_keys()
        client.load_host_keys(str(Path.home() / ".ssh" / "known_hosts"))
        client.set_missing_host_key_policy(self.module.RejectPolicy())
        client.connect(HOST, port=PORT, username=USER, password=self.password,
                       look_for_keys=False, allow_agent=False, timeout=20,
                       auth_timeout=20, banner_timeout=20)
        client.get_transport().set_keepalive(15)
        self.client = client

    def close(self):
        if self.client is not None:
            self.client.close()
            self.client = None

    def retry(self, operation):
        for attempt in range(5):
            try:
                if self.client is None or not self.client.get_transport().is_active():
                    self.connect()
                return operation()
            except (EOFError, socket.timeout, ConnectionError, self.module.SSHException):
                self.close()
                if attempt == 4:
                    raise RuntimeError("SSH_TRANSPORT_RECONNECT_EXHAUSTED") from None
                time.sleep(5)

    def python(self, body, *, retry=True):
        def operation():
            # Paramiko starts a non-login shell; reproduce the verified production PATH.
            command = ("PATH=/root/miniconda3/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH "
                       "/root/miniconda3/bin/python -c " + shlex.quote(body))
            _, stdout, stderr = self.client.exec_command(command, timeout=90)
            out, err = stdout.read(), stderr.read()
            require(stdout.channel.recv_exit_status() == 0,
                    "REMOTE_CHECK_FAILED:" + err.decode("utf-8", "replace")[-1600:])
            return json.loads(out)
        return self.retry(operation) if retry else operation()

    def json(self, path, optional=False):
        require(path.startswith(BASE + "/"), "REMOTE_PATH_OUT_OF_SCOPE")
        return self.python("import json,pathlib; p=pathlib.Path(" + repr(path) + "); "
                           + "print(json.dumps(json.loads(p.read_text()) if p.exists() else None))") if optional else self.python(
                               "import json,pathlib; print(json.dumps(json.loads(pathlib.Path(" + repr(path) + ").read_text())))")

    def download(self, remote_path, local_path, expected_sha=None, expected_bytes=None):
        require(remote_path.startswith(EXPORTS + "/"), "DOWNLOAD_OUT_OF_SCOPE")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if local_path.exists() and expected_sha:
            require(file_sha(local_path) == expected_sha, "CACHED_DOWNLOAD_CHANGED")
            if expected_bytes is not None:
                require(local_path.stat().st_size == expected_bytes, "CACHED_DOWNLOAD_SIZE_CHANGED")
            return
        temporary = local_path.with_name(local_path.name + ".download." + uuid.uuid4().hex)
        def operation():
            # Interrupted transport files stay visible; each retry gets a fresh path.
            attempt_path = temporary.with_name(temporary.name + "." + uuid.uuid4().hex)
            with self.client.open_sftp() as sftp:
                sftp.get(remote_path, str(attempt_path))
            return attempt_path
        received = self.retry(operation)
        if expected_sha:
            require(file_sha(received) == expected_sha, "DOWNLOAD_HASH_FAILED")
        if expected_bytes is not None:
            require(received.stat().st_size == expected_bytes, "DOWNLOAD_SIZE_FAILED")
        if local_path.exists():
            require(file_sha(local_path) == file_sha(received), "DOWNLOAD_OVERWRITE_CONFLICT")
        else:
            received.rename(local_path)

    def write_once(self, path, value):
        require(path.startswith(BASE + "/"), "REMOTE_WRITE_OUT_OF_SCOPE")
        body = """import json,pathlib
p=pathlib.Path(PATH); value=VALUE
if p.exists():
 assert json.loads(p.read_text())==value, 'REMOTE_IMMUTABLE_CONFLICT'
else:
 with p.open('x') as f: json.dump(value,f,indent=2); f.write('\\n')
print(json.dumps({'status':'VERIFIED'}))
""".replace("PATH", repr(path)).replace("VALUE", repr(value))
        # Re-reading an uncertain successful write is safe; no destructive overwrite.
        return self.python(body)


class Coordinator:
    def __init__(self, repo, remote):
        self.repo, self.remote = repo.resolve(), remote
        self.output = self.repo / "results/physical_identity_mainline_pilot"
        self.transport = self.output / ".transport"
        self.transport.mkdir(exist_ok=True)
        self.journal = self.transport / "coordinator_journal.jsonl"

    def event(self, status, case=None, **extra):
        record = dict(utc=datetime.now(timezone.utc).isoformat(), status=status, **extra)
        if case:
            record["case"] = case
        with self.journal.open("ab") as stream:
            stream.write(json.dumps(record, allow_nan=False).encode() + b"\n")
        print(json.dumps(record), flush=True)

    def git(self, *args, data=None, check=True):
        command = ["git", "-c", "safe.directory=" + self.repo.as_posix(), *args]
        result = subprocess.run(command, cwd=self.repo, input=data, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=300)
        require(not check or result.returncode == 0,
                "GIT_OPERATION_FAILED:" + " ".join(args[:2]) + ":" + result.stderr.decode("utf-8", "replace")[-1200:])
        return result

    def clean_index(self):
        require(not self.git("diff", "--cached", "--name-only", "-z").stdout, "UNRELATED_OR_PREEXISTING_STAGED_FILES")

    def preflight(self):
        self.clean_index()
        design = read(self.output / "design.json")
        require(design["fingerprint"] == DESIGN, "LOCAL_DESIGN_CHANGED")
        require(tuple(x["key"] for x in design["scientific"]["membership"]) == CASES, "SIX_CASE_MEMBERSHIP_CHANGED")
        require(digest(json.dumps(design["scientific"], sort_keys=True, separators=(",", ":"),
                                  allow_nan=False).encode()) == DESIGN, "LOCAL_DESIGN_CONTENT_CHANGED")
        self.git("merge-base", "--is-ancestor", PREPARED_COMMIT, "HEAD")
        source_manifest = read(self.output / "source_manifest.json")
        expected = {name: item["sha256"] for name, item in source_manifest["files"].items()}
        for name, value in expected.items():
            require(file_sha(self.repo / name) == value, "LOCAL_FROZEN_SOURCE_CHANGED:" + name)
        body = """import json,hashlib,pathlib
base=pathlib.Path(BASE); expected=EXPECTED
assert json.loads((base/'results/design.json').read_text())['fingerprint']==DESIGN
assert hashlib.sha256((base/'results/design.json').read_bytes()).hexdigest()==DESIGN_SHA
assert json.loads((base/'source/source_manifest.json').read_text())==MANIFEST
for name,value in expected.items():
 assert hashlib.sha256((base/'source'/name).read_bytes()).hexdigest()==value, name
print(json.dumps({'status':'VERIFIED','files':len(expected)}))
""".replace("DESIGN_SHA", repr(file_sha(self.output / "design.json"))).replace("DESIGN", repr(DESIGN))
        body = body.replace("EXPECTED", repr(expected)).replace("MANIFEST", repr(source_manifest)).replace("BASE", repr(BASE))
        self.remote.python(body)
        self.event("PREFLIGHT_PASS")

    def command(self, case):
        return ["python", SOURCE + "/analysis/run_physical_identity_mainline_pilot.py", "train-case",
                "--snapshot", SOURCE, "--output", OUTPUT, "--case", case]

    def process(self, pid, command):
        return self.remote.python("""import pathlib,json
p=pathlib.Path('/proc')/str(PID)/'cmdline'
try: args=p.read_bytes().split(b'\\0')
except (FileNotFoundError,ProcessLookupError): args=[]
actual=[x.decode() for x in args if x]
if actual: assert actual==COMMAND, 'PROCESS_IDENTITY_CHANGED'
print(json.dumps({'alive':bool(actual)}))
""".replace("PID", repr(pid)).replace("COMMAND", repr(command)))

    def launch(self, case):
        path = JOBS + "/" + case + ".launch.json"
        launch = self.remote.json(path, optional=True)
        if launch is None:
            require(case != CASES[0], "FIRST_LAUNCH_RECORD_MISSING_DO_NOT_DUPLICATE")
            prior = CASES[:CASES.index(case)]
            for previous in prior:
                self.verify_ack(previous)
            body = """import pathlib,json,subprocess,shutil,datetime,os
jobs=pathlib.Path(JOBS); path=jobs/NAME
if path.exists():
 print(path.read_text()); raise SystemExit
assert shutil.disk_usage(OUTPUT).free >= 1.4*1024**3, 'INSUFFICIENT_ROOT_SPACE'
active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).decode().strip()
assert not active, 'GPU_PROCESS_ACTIVE'
util=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits']).decode().split()
assert util and all(float(x)==0 for x in util), 'GPU_NOT_IDLE'
for process_id in subprocess.check_output(['ps','-eo','pid=']).split():
 try:
  argv=[x.decode() for x in (pathlib.Path('/proc')/process_id.decode()/'cmdline').read_bytes().split(b'\\0') if x]
 except (FileNotFoundError,ProcessLookupError,PermissionError):
  continue
 assert not (RUNNER in argv and 'train-case' in argv), 'PILOT_FIT_ACTIVE'
reservation=jobs/(CASE+'.launch_reserved.json')
with reservation.open('x') as f: json.dump({'case':CASE,'command':COMMAND},f)
with (jobs/(CASE+'.log')).open('xb') as log:
 proc=subprocess.Popen(COMMAND,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
value=dict(case=CASE,pid=proc.pid,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=COMMAND,
 prepared_design_commit=COMMIT,design_fingerprint=DESIGN,status='LAUNCHED_NOT_YET_COMPLETE')
with path.open('x') as f: json.dump(value,f,indent=2); f.write('\\n')
print(json.dumps(value))
"""
            replacements = {"RUNNER": repr(self.command(case)[1]), "JOBS": repr(JOBS), "NAME": repr(case + ".launch.json"),
                            "OUTPUT": repr(OUTPUT), "CASE": repr(case), "COMMAND": repr(self.command(case)),
                            "COMMIT": repr(PREPARED_COMMIT), "DESIGN": repr(DESIGN)}
            for key, val in replacements.items():
                body = body.replace(key, val)
            # A reserved-but-unrecorded launch stops future attempts rather than risking duplication.
            launch = self.remote.python(body)
        require(launch["case"] == case and launch["command"] == self.command(case) and
                launch["prepared_design_commit"] == PREPARED_COMMIT and launch["design_fingerprint"] == DESIGN,
                "LAUNCH_PROVENANCE_CHANGED")
        if case == CASES[0]:
            require(launch["pid"] == 115253, "FIRST_LAUNCH_PID_CHANGED")
        immutable(self.output / "cases" / case / "launch.json", encoded(launch))
        return launch

    def wait_fit(self, case, launch):
        deadline = datetime.fromisoformat(launch["started_utc"]).timestamp() + 7200
        while True:
            failure = self.remote.json(OUTPUT + "/failure.json", optional=True)
            require(failure is None, "FROZEN_RUNNER_REPORTED_FAILURE")
            completed = self.remote.json(OUTPUT + "/cases/" + case + "/training_complete.json", optional=True)
            alive = self.process(launch["pid"], launch["command"])["alive"]
            if not alive and completed is None:
                completed = self.remote.json(OUTPUT + "/cases/" + case + "/training_complete.json", optional=True)
            if completed is not None:
                require(completed["case"] == case and completed["fingerprint"] == DESIGN and
                        completed["status"] == "NORMAL_FINITE_COMPLETE", "COMPLETION_BINDING_FAILED")
                if not alive:
                    self.event("FIT_COMPLETE_REVIEW_NEXT", case)
                    return
            require(alive, "FIT_EXITED_WITHOUT_COMPLETION")
            require(time.time() < deadline, "FIT_TWO_HOUR_LIMIT_REACHED")
            status = self.remote.json(OUTPUT + "/status.json", optional=True)
            require(status is None or status.get("case") == case, "ACTIVE_CASE_STATUS_CHANGED")
            self.event("TRAINING_PROCESS_ALIVE", case)
            time.sleep(30)

    def review_export(self, case):
        """One detached operational wrapper records reviewer exit; reconnects never duplicate it."""
        command = ["python", SOURCE + "/analysis/review_physical_identity_mainline_pilot.py", "--output", OUTPUT,
                   "--snapshot", SOURCE, "--case", case, "--export-dir", EXPORTS]
        exit_path, launch_path = JOBS + "/" + case + ".review.exit.json", JOBS + "/" + case + ".review.launch.json"
        wrapper = ("import subprocess,pathlib,json; command=" + repr(command) + "; "
                   + "r=subprocess.run(command); p=pathlib.Path(" + repr(exit_path) + "); "
                   + "p.write_text(json.dumps({'returncode':r.returncode})+'\\n')")
        wrapper_command = ["python", "-c", wrapper]
        body = """import json,pathlib,subprocess,datetime
p=pathlib.Path(PATH)
if p.exists(): print(p.read_text()); raise SystemExit
reservation=p.with_suffix('.reserved')
with reservation.open('x') as f: f.write('reserved\\n')
with pathlib.Path(LOG).open('xb') as log:
 proc=subprocess.Popen(COMMAND,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
v=dict(pid=proc.pid,command=COMMAND,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
with p.open('x') as f: json.dump(v,f)
print(json.dumps(v))
""".replace("PATH", repr(launch_path)).replace("LOG", repr(JOBS + "/" + case + ".review.log")).replace("COMMAND", repr(wrapper_command))
        launch = self.remote.python(body)
        require(launch["command"] == wrapper_command, "REVIEW_COMMAND_CHANGED")
        deadline = datetime.fromisoformat(launch["started_utc"]).timestamp() + 1800
        while True:
            result = self.remote.json(exit_path, optional=True)
            if result is not None:
                require(result["returncode"] == 0, "INDEPENDENT_REVIEW_FAILED")
                return
            if not self.process(launch["pid"], launch["command"])["alive"]:
                result = self.remote.json(exit_path, optional=True)
                require(result is not None and result["returncode"] == 0, "REVIEW_EXITED_WITHOUT_SUCCESS_RECEIPT")
                return
            require(time.time() < deadline, "REVIEW_TIME_LIMIT_REACHED")
            self.event("INDEPENDENT_PROCESS_REVIEW_RUNNING", case)
            time.sleep(30)

    def transfer(self, case):
        receipt_path = self.transport / (case + ".receipt.json")
        self.remote.download(EXPORTS + "/" + case + ".receipt.json", receipt_path)
        receipt = read(receipt_path)
        require(receipt["case"] == case and receipt["design_fingerprint"] == DESIGN and
                receipt["no_source_artifacts_deleted"] is True, "EXPORT_RECEIPT_BINDING_FAILED")
        require(receipt["archive_path"] == EXPORTS + "/" + case + ".tar.gz", "EXPORT_PATH_CHANGED")
        require(0 < receipt["archive_bytes"] < 256 * 1024**2, "EXPORT_ARCHIVE_SIZE_UNEXPECTED")
        archive = self.transport / (case + ".tar.gz")
        self.remote.download(receipt["archive_path"], archive, receipt["archive_sha256"], receipt["archive_bytes"])
        prefix = "cases/" + case + "/"
        manifest = receipt["files"]
        require(COMMON <= set(manifest), "COMMON_EXPORT_FILES_MISSING")
        require({prefix + n for n in REQUIRED_CASE_FILES} <= set(manifest), "CASE_EXPORT_FILES_MISSING")
        for name in manifest:
            safe_member(self.output, name)
            require(name in COMMON or name.startswith(prefix), "OTHER_CASE_OR_UNEXPECTED_COMMON_EXPORT")
            require(name in COMMON or Path(name).suffix in TEXT_SUFFIXES or name == prefix + "evidence.npz", "UNEXPECTED_BINARY_EXPORT")
        # Validate every member before writing any extracted output.
        with tarfile.open(archive, "r:gz") as tar:
            members = tar.getmembers()
            require(len(members) == len(manifest) and {m.name for m in members} == set(manifest), "ARCHIVE_MEMBERSHIP_FAILED")
            require(sum(m.size for m in members) < 512 * 1024**2, "ARCHIVE_EXPANSION_LIMIT")
            for member in members:
                require(member.isfile() and not member.issym() and not member.islnk() and
                        member.size < 64 * 1024**2, "NONREGULAR_OR_OVERSIZED_ARCHIVE_MEMBER")
                data = tar.extractfile(member).read()
                require(digest(data) == manifest[member.name], "ARCHIVE_MEMBER_HASH_FAILED")
                path = safe_member(self.output, member.name)
                if member.name in COMMON:
                    require(path.is_file() and file_sha(path) == manifest[member.name], "COMMON_FILE_CHANGED")
                elif path.exists():
                    require(file_sha(path) == manifest[member.name], "EXISTING_CASE_FILE_CHANGED")
            for member in members:
                if member.name not in COMMON:
                    immutable(safe_member(self.output, member.name), tar.extractfile(member).read())
        case_dir = self.output / "cases" / case
        review = read(case_dir / "case_review.json")
        require(review["status"] == "PASS" and review["case"] == case and review["design_fingerprint"] == DESIGN,
                "INDEPENDENT_CASE_REVIEW_NOT_PASS")
        require(file_sha(case_dir / "training_complete.json") == receipt["training_complete_sha256"] ==
                review["training_complete_sha256"], "TRAINING_COMPLETE_HASH_CHANGED")
        expected = dict(review["compact_files"])
        expected[prefix + "case_review.json"] = file_sha(case_dir / "case_review.json")
        require(expected == manifest, "REVIEW_EXPORT_MEMBERSHIP_CHANGED")
        require(review["normal_completion"] is True and review["all_outputs_losses_finite"] is True and
                review["runtime_and_source_bindings_verified"] is True and review["cleanup_performed"] is False,
                "PROCESS_REVIEW_GATES_FAILED")
        immutable(case_dir / "export_receipt.json", receipt_path.read_bytes())
        transfer = dict(status="VERIFIED_DOWNLOAD_PENDING_CASE_GIT", case=case, design_fingerprint=DESIGN,
                        archive_sha256=receipt["archive_sha256"], training_complete_sha256=receipt["training_complete_sha256"],
                        verified_files=manifest, exact_membership_verified=True, source_review_status="PASS",
                        evidence_storage=str(case_dir / "evidence.npz"), evidence_sha256=manifest[prefix + "evidence.npz"],
                        unique_artifacts_retained=True, outcome_analysis="DEFERRED_UNTIL_CAL_SEAL", cleanup_performed=False)
        immutable(case_dir / "transfer_review.json", encoded(transfer))
        paths = [safe_member(self.output, name) for name in manifest if name.startswith(prefix) and Path(name).suffix in TEXT_SUFFIXES]
        paths += [case_dir / "export_receipt.json", case_dir / "transfer_review.json", case_dir / "launch.json"]
        self.event("EXACT_CASE_DOWNLOAD_VERIFIED", case, verified_file_count=len(manifest))
        return receipt, paths

    def blob(self, path):
        return self.git("hash-object", "--stdin", data=path.read_bytes()).stdout.strip().decode()

    def verify_commit_files(self, case, commit, manifest):
        prefix = "cases/" + case + "/"
        for name in REQUIRED_CASE_FILES:
            relative = "results/physical_identity_mainline_pilot/" + prefix + name
            data = self.git("show", commit + ":" + relative).stdout
            require(digest(data) == manifest[prefix + name], "PUSHED_CASE_CONTENT_CHANGED:" + name)

    def save_case(self, case, receipt, paths):
        self.clean_index()
        relatives = sorted({p.relative_to(self.repo).as_posix() for p in paths})
        self.git("add", "--", *relatives)
        staged = set(self.git("diff", "--cached", "--name-only", "-z").stdout.decode().rstrip("\x00").split("\x00")) - {""}
        require(staged <= set(relatives), "UNRELATED_STAGED_FILES_APPEARED")
        for relative in relatives:
            require(self.git("rev-parse", ":" + relative).stdout.strip().decode() == self.blob(self.repo / relative),
                    "GIT_INDEX_BYTES_CHANGED:" + relative)
        self.git("diff", "--cached", "--check")
        if staged:
            self.git("commit", "-m", "Record audited physical identity pilot case " + case)
        commit = self.git("rev-parse", "HEAD").stdout.strip().decode()
        self.verify_commit_files(case, commit, receipt["files"])
        self.git("push", "origin", "HEAD:" + BRANCH)
        remote = self.git("ls-remote", "origin", BRANCH).stdout.decode().split()
        require(remote and remote[0] == commit, "PUSHED_COMMIT_NOT_VERIFIED")
        self.verify_commit_files(case, commit, receipt["files"])
        self.clean_index()
        self.event("CASE_COMMITTED_AND_PUSHED", case, commit=commit)
        return commit

    def verify_ack(self, case):
        ack = self.remote.json(OUTPUT + "/cases/" + case + "/handoff_ack.json")
        require(ack["status"] == "VERIFIED_DOWNLOADED_AND_PUSHED" and ack["case"] == case, "ACK_INVALID")
        case_dir = self.output / "cases" / case
        require(file_sha(case_dir / "training_complete.json") == ack["training_complete_sha256"], "ACK_LOCAL_CASE_CHANGED")
        receipt = read(case_dir / "export_receipt.json")
        require(receipt["archive_sha256"] == ack["archive_sha256"], "ACK_ARCHIVE_CHANGED")
        self.verify_commit_files(case, ack["pushed_commit"], receipt["files"])
        tip = self.git("ls-remote", "origin", BRANCH).stdout.decode().split()
        require(bool(tip), "AUTHORIZED_REMOTE_BRANCH_MISSING")
        self.git("merge-base", "--is-ancestor", ack["pushed_commit"], tip[0])
        remote_complete = self.remote.python("import pathlib,hashlib,json; print(json.dumps(hashlib.sha256(pathlib.Path(" +
                                            repr(OUTPUT + "/cases/" + case + "/training_complete.json") + ").read_bytes()).hexdigest()))")
        require(remote_complete == ack["training_complete_sha256"], "ACK_REMOTE_CASE_CHANGED")
        immutable(case_dir / "handoff_ack.json", encoded(ack))
        return ack

    def run(self):
        self.preflight()
        for case in CASES:
            ack = self.remote.json(OUTPUT + "/cases/" + case + "/handoff_ack.json", optional=True)
            if ack is not None:
                self.verify_ack(case)
                self.event("EXISTING_PUSHED_HANDOFF_VERIFIED", case)
                continue
            launch = self.launch(case)
            self.wait_fit(case, launch)
            self.review_export(case)
            receipt, paths = self.transfer(case)
            commit = self.save_case(case, receipt, paths)
            ack = dict(status="VERIFIED_DOWNLOADED_AND_PUSHED", case=case,
                       training_complete_sha256=receipt["training_complete_sha256"], pushed_commit=commit,
                       archive_sha256=receipt["archive_sha256"], design_fingerprint=DESIGN)
            self.remote.write_once(OUTPUT + "/cases/" + case + "/handoff_ack.json", ack)
            immutable(self.output / "cases" / case / "handoff_ack.json", encoded(ack))
            self.verify_ack(case)
            self.event("CASE_HANDOFF_ACK_VERIFIED", case)
        self.event("ALL_SIX_HANDOFFS_COMPLETE", scoring_started=False, deletion_performed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--inspect", action="store_true", help="Print fixed scope only; no network, password, Git mutation or training.")
    args = parser.parse_args()
    if args.inspect:
        print(json.dumps(dict(cases=CASES, host=HOST, port=PORT, remote_root=BASE, design_fingerprint=DESIGN,
                              prepared_commit=PREPARED_COMMIT, target_branch=BRANCH, scoring=False, deletion=False), indent=2))
        return
    require(sys.stdin.isatty(), "INTERACTIVE_PASSWORD_TERMINAL_REQUIRED")
    password = getpass.getpass("Westc SSH password (memory only): ")
    remote = Remote(password)
    password = None
    coordinator = Coordinator(args.repo, remote)
    try:
        coordinator.run()
    except BaseException as exc:
        coordinator.event("STOPPED_REQUIRES_REVIEW", error_type=type(exc).__name__, error=str(exc)[:1800])
        raise
    finally:
        remote.close()
        remote.password = None


if __name__ == "__main__":
    main()
