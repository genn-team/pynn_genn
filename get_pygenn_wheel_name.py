from sys import argv
from wheel import pep425tags as w

assert len(argv) == 3

print("cuda%s-pygenn-%s-%s%s-%s-%s.whl" %
      (argv[1], argv[2], w.get_abbr_impl(), w.get_impl_ver(),
       w.get_abi_tag(), w.get_platform()))
