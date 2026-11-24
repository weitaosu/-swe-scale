"""Select userspace networking explicitly for this project's Podman requests."""
import os


def container_options(options):
    result=dict(options)
    if os.environ.get('SWE_STUDY_CONTAINER_ENGINE')=='podman-hpc':
        mode=result.get('network_mode')
        if mode in (None,'','default','bridge'):
            result['network_mode']='slirp4netns'
        elif not isinstance(mode,str) or not mode.startswith('slirp4netns'):
            raise ValueError('Project Podman containers require slirp4netns networking')
    return result
