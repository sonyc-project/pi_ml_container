import os


def filewatch(*paths, daemonize=True):
    import pyinotify
    def inner(on_create):
        wm = pyinotify.WatchManager()
        notifier = pyinotify.Notifier(wm)
        for path in paths:
            wm.add_watch(path, pyinotify.IN_CREATE)
        return lambda: notifier.loop(daemonize=daemonize, callback=on_create)
    return inner


def get_output_path(filepath, suffix=None, output_path=None):
    fbase, ext = os.path.splitext(os.path.basename(filepath))

    output_path = output_path or os.path.dirname(filepath)
    os.makedirs(output_path, exist_ok=True)
    ext = ('_' + suffix if suffix and suffix[0] not in ('.', '_')
           else suffix or ('_out' + ext))

    return os.path.join(output_path, fbase + ext)
