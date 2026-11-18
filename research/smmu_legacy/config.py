"""Where pravaha keeps its per-machine settings.

Two small files, shared by the C library and the Python package:

    unit              one line: the confirmed counter unit, e.g. smmuv3_pmcg_13002
    calibration.json  bytes per transaction by copy size, and the idle background

They are looked up in this order (the first that exists wins):

    $PRAVAHA_CONFIG_DIR
    $XDG_CONFIG_HOME/pravaha     (usually ~/.config/pravaha)
    ~/.config/pravaha
    ~SUDO_USER/.config/pravaha   (under sudo: the settings of the user who ran sudo)
    /etc/pravaha                 (machine wide)

``pravaha setup`` writes to the first directory of that list
(or to /etc/pravaha with --system). The environment variable PRAVAHA_UNIT
overrides the unit file.
"""

import os


def search_dirs():
    dirs = []
    if os.environ.get("PRAVAHA_CONFIG_DIR"):
        dirs.append(os.environ["PRAVAHA_CONFIG_DIR"])
    if os.environ.get("XDG_CONFIG_HOME"):
        dirs.append(os.path.join(os.environ["XDG_CONFIG_HOME"], "pravaha"))
    dirs.append(os.path.join(os.path.expanduser("~"), ".config", "pravaha"))
    sudo_user = os.environ.get("SUDO_USER")
    if sudo_user and os.geteuid() == 0:          # under sudo: also the settings of the user who ran it
        try:
            import pwd
            dirs.append(os.path.join(pwd.getpwnam(sudo_user).pw_dir, ".config", "pravaha"))
        except KeyError:
            pass
    dirs.append("/etc/pravaha")
    seen, out = set(), []
    for d in dirs:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def find_file(name):
    """Full path of the first existing config file with this name, or None."""
    for d in search_dirs():
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return None


def write_dir(system=False):
    """Directory to write settings to (created if needed)."""
    d = "/etc/pravaha" if system else search_dirs()[0]
    os.makedirs(d, exist_ok=True)
    return d


def saved_unit():
    """The unit chosen earlier: PRAVAHA_UNIT, else the unit file, else None."""
    if os.environ.get("PRAVAHA_UNIT"):
        return os.environ["PRAVAHA_UNIT"].strip()
    p = find_file("unit")
    if p:
        with open(p) as f:
            name = f.readline().strip()
        return name or None
    return None


def save_unit(unit, system=False):
    path = os.path.join(write_dir(system), "unit")
    with open(path, "w") as f:
        f.write(unit + "\n")
    return path
