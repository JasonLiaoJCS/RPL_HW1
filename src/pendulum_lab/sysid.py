"""CLI adapter for the supplied identification wizard."""
from .sysid_wizard import configure_parser, execute


def add_parser(sub):
    parser = sub.add_parser("sysid", help="tare, collect, fit and check your assembled rig")
    configure_parser(parser)
