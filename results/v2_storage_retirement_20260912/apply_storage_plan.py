"""Apply one audited V2 storage plan; retain every bound checkpoint byte."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main(args):
    assert re.fullmatch('[0-9a-f]{40}', args.git_plan_commit)
    assert sha(args.plan) == args.plan_sha256
    plan = read(args.plan)
    source = Path('/root/autodl-tmp/lipiddeconv/results/ce_identity_spatial_v2')
    retained = Path('/root/v2_retained_runtime_20260912')
    exports = Path('/root/v58_jobs/ce_identity_spatial_v2_exports')
    assert plan['source_root'] == str(source)
    assert plan['retained_storage_root'] == str(retained)
    assert source.resolve() == source and retained.resolve() == retained
    assert exports.resolve() == exports and source.stat().st_dev != retained.stat().st_dev
    assert len(plan['relocations']) == 36 and len(plan['remove_transport_archives']) == 6
    backup = read(args.backup)
    assert backup['status'] == 'PASS'
    assert backup['pushed_case_snapshot'] == plan['pushed_case_snapshot']
    archive_backups = {r['case']: r for r in backup['archives']}
    assert len(archive_backups) == 6
    assert not args.receipt.exists()
    for item in plan['relocations']:
        src, dst = Path(item['source']), Path(item['destination'])
        assert src.is_relative_to(source / 'cases') and src.parent.name == 'training'
        assert src.suffix == '.pth' and src.resolve() == src and not src.is_symlink()
        assert dst == retained / src.relative_to(source) and dst.resolve() == dst
        assert dst.is_file() and not dst.is_symlink()
        assert sha(src) == sha(dst) == item['sha256']
        assert src.stat().st_size == dst.stat().st_size == item['bytes']
    for item in plan['remove_transport_archives']:
        p = Path(item['path'])
        assert p.parent == exports and p.name == item['case'] + '.tar.gz'
        assert p.resolve() == p and not p.is_symlink()
        b = archive_backups[item['case']]
        assert sha(p) == item['sha256'] == b['archive_sha256']
        assert p.stat().st_size == item['bytes'] == b['bytes']
    for path, wanted in plan['protected_hashes'].items():
        assert sha(Path(path)) == wanted, path
    result = dict(status='IN_PROGRESS', started_utc=datetime.now(timezone.utc).isoformat(),
                  plan_sha256=args.plan_sha256, git_plan_commit=args.git_plan_commit,
                  pushed_case_snapshot=plan['pushed_case_snapshot'],
                  apply_source_sha256=sha(Path(__file__)),
                  local_backup_verification_sha256=sha(args.backup),
                  free_before=dict(data=shutil.disk_usage(source)._asdict(),
                                   root=shutil.disk_usage(retained)._asdict()),
                  relocated=[], removed_archives=[], unique_checkpoint_bytes_deleted=0,
                  scientific_contract_changed=False, training_or_scoring_interrupted=False)

    def save():
        tmp = args.receipt.with_suffix('.partial.json')
        tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        os.replace(tmp, args.receipt)

    save()
    try:
        for item in plan['relocations']:
            src, dst = Path(item['source']), Path(item['destination'])
            assert src.resolve() == src and not src.is_symlink()
            assert sha(src) == sha(dst) == item['sha256']
            link = src.with_name(src.name + '.verified_storage_link')
            assert not link.exists() and not link.is_symlink()
            link.symlink_to(dst)
            assert sha(link) == item['sha256']
            os.replace(link, src)
            assert src.is_symlink() and src.resolve() == dst and sha(src) == item['sha256']
            result['relocated'].append(item)
            save()
        # All original bindings still resolve before redundant archives are retired.
        for path, wanted in plan['protected_hashes'].items():
            assert sha(Path(path)) == wanted, path
        for item in plan['remove_transport_archives']:
            p = Path(item['path'])
            assert p.resolve() == p and not p.is_symlink() and sha(p) == item['sha256']
            p.unlink()
            assert not p.exists()
            result['removed_archives'].append(item)
            save()
        for path, wanted in plan['protected_hashes'].items():
            assert sha(Path(path)) == wanted, path
        result.update(status='COMPLETE', finished_utc=datetime.now(timezone.utc).isoformat(),
                      protected_hashes_verified=len(plan['protected_hashes']),
                      data_disk_bytes_released=sum(x['bytes'] for x in result['relocated']),
                      redundant_archive_bytes_deleted=sum(x['bytes'] for x in result['removed_archives']),
                      retained_storage=str(retained),
                      free_after=dict(data=shutil.disk_usage(source)._asdict(),
                                      root=shutil.disk_usage(retained)._asdict()))
        save()
    except BaseException as exc:
        result.update(status='PARTIAL_REQUIRES_REVIEW', error=repr(exc))
        save()
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--plan-sha256', required=True)
    p.add_argument('--git-plan-commit', required=True)
    p.add_argument('--backup', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    main(p.parse_args())
