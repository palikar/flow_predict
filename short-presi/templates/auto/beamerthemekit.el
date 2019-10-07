(TeX-add-style-hook
 "beamerthemekit"
 (lambda ()
   (TeX-run-style-hooks
    "templates/beamerthemekitbase")
   (LaTeX-add-lengths
    "kitbottom"))
 :latex)

