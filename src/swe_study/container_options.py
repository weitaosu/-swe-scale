"""Explicit networking and Docker-compatible image pulls for project Podman."""
import os


def pull_reference(image):
    """Preserve Docker Hub defaults instead of Podman's site search registries."""
    if os.environ.get('SWE_STUDY_CONTAINER_ENGINE')!='podman-hpc': return image
    if not isinstance(image,str) or not image or image.startswith('-') or any(c.isspace() for c in image):
        raise ValueError('Expected a container image reference')
    if '://' in image or image.startswith('sha256:'): return image
    if '/' not in image: return 'docker.io/library/'+image
    registry=image.split('/',1)[0]
    if '.' in registry or ':' in registry or registry=='localhost': return image
    return 'docker.io/'+image


def container_options(options):
    result=dict(options)
    if os.environ.get('SWE_STUDY_CONTAINER_ENGINE')=='podman-hpc':
        mode=result.get('network_mode')
        if mode in (None,'','default','bridge'):
            result['network_mode']='slirp4netns'
        elif not isinstance(mode,str) or not mode.startswith('slirp4netns'):
            raise ValueError('Project Podman containers require slirp4netns networking')
    return result
