# cli

The command line, one subcommand per thing a reader of the article is likely to
want from a terminal: `reproduce`, `simulate`, `fit`, `test-memoryless`,
`test-periodicity`, `cite` and `params`. Each prints its own help with
`msrelapse <subcommand> --help`.

This module composes the rest of the package and computes nothing of its own.
Every number it prints comes from another module, and an error raised by any of
them travels out with its own message rather than being turned into an exit
code here. The one exception is `reproduce`, which exits 1 when a judged row of
the closing table falls outside its tolerance.

::: msrelapse.cli
