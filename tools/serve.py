"""Start the site with a listen queue deep enough for the dashboard.

The BI page asks for eighteen endpoints at once. Django's development server
queues ten pending connections and the operating system refuses the rest, which
reaches the browser as a scattering of 502s from whichever panels lost the race.
Raising the queue costs nothing and leaves the rest of runserver alone.
"""

import os
import sys

from django.core.servers.basehttp import WSGIServer

WSGIServer.request_queue_size = 256

if __name__ == "__main__":
    project = sys.argv[1]
    port = sys.argv[2] if len(sys.argv) > 2 else "8000"
    sys.path.insert(0, project)
    os.chdir(project)
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "market.settings")

    from django.core.management import execute_from_command_line

    execute_from_command_line(
        ["manage.py", "runserver", port, "--insecure", "--noreload"]
    )
