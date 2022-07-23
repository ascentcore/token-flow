import os
import sys

from commands import digest, dev


commands = {
    'digest': digest,
    'dev': dev
}

print(sys.argv)

if len(sys.argv) < 2:
    print("Usage: tokenflow.py <command> [<args>]")
    print("Commands:")
    print(" digest <path>")
    sys.exit(1)

command = sys.argv[1]

if command in commands.keys():
    commands[command].execute(sys.argv[2:])
else:
    print(f'Unkown command {command}')
    sys.exit(1)