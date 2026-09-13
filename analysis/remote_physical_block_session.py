"""Interactive authenticated transport; credentials stay only in memory."""
import getpass
import hashlib
import json
from pathlib import Path
import sys
import paramiko

HOST = 'connect.westc.seetacloud.com'
PORT = 55786


def main():
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    known = Path.home() / '.ssh/known_hosts'
    if known.exists():
        client.load_host_keys(str(known))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    password = getpass.getpass('Remote password: ')
    client.connect(HOST, port=PORT, username='root', password=password,
                   look_for_keys=False, allow_agent=False, timeout=15)
    del password
    client.get_transport().set_keepalive(25)
    print('AUTHENTICATED_READY', flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request['op'] == 'close':
                break
            if request['op'] == 'run':
                stdin, stdout, stderr = client.exec_command(request['command'], timeout=request.get('timeout', 300))
                stdin.close()
                out, err = stdout.read().decode(), stderr.read().decode()
                print(json.dumps(dict(exit_code=stdout.channel.recv_exit_status(), stdout=out, stderr=err)), flush=True)
            elif request['op'] in ('put', 'get'):
                with client.open_sftp() as sftp:
                    for entry in request['files']:
                        local, remote = Path(entry['local']), entry['remote']
                        if request['op'] == 'put':
                            with local.open('rb') as source, sftp.open(remote, 'wb') as target:
                                while chunk := source.read(1024 * 1024):
                                    target.write(chunk)
                        else:
                            local.parent.mkdir(parents=True, exist_ok=True)
                            if local.exists():
                                raise RuntimeError('Existing download preserved: ' + str(local))
                            sftp.get(remote, str(local))
                        print(json.dumps(dict(transferred=request['op'], local=str(local), remote=remote,
                                              sha256=hashlib.sha256(local.read_bytes()).hexdigest())), flush=True)
            else:
                raise ValueError('Unknown operation')
        except Exception as exc:
            print(json.dumps(dict(error=type(exc).__name__, message=str(exc))), flush=True)
        print('REQUEST_COMPLETE', flush=True)
    client.close()


if __name__ == '__main__':
    main()
