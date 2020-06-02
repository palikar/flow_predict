(TeX-add-style-hook
 "meth"
 (lambda ()
   (TeX-run-style-hooks
    "latex2e"
    "standalone"
    "standalone10"
    "tikz"
    "pgfplots"
    "tkz-euclide")
   (TeX-add-symbols
    "BeforeLight"))
 :latex)

