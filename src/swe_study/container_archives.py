"""Upload task files using container ownership rather than unmapped host IDs."""
import io
from pathlib import PurePosixPath
import tarfile


def container_owner(member):
    member.uid=member.gid=0
    member.uname=member.gname='root'
    member.pax_headers={key:value for key,value in member.pax_headers.items()
                        if key not in {'uid','gid','uname','gname'}}
    return member


def copy_to_container(container,src,dst):
    """Podman-compatible equivalent of the pinned grader's file-copy helper."""
    dst=PurePosixPath(dst)
    if not dst.is_absolute() or dst.name in {'','..'} or '..' in dst.parts:
        raise ValueError('Expected an absolute container file destination')
    payload=io.BytesIO()
    with tarfile.open(fileobj=payload,mode='w') as archive:
        archive.add(src,arcname=dst.name,filter=container_owner)
    result=container.exec_run(['mkdir','-p',str(dst.parent)])
    if result.exit_code:
        raise RuntimeError('Cannot create container upload directory: '+str(dst.parent))
    if not container.put_archive(str(dst.parent),payload.getvalue()):
        raise RuntimeError('Container archive upload failed: '+str(dst))
