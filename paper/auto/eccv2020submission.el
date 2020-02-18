(TeX-add-style-hook
 "eccv2020submission"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("hyperref" "pagebackref")))
   (TeX-run-style-hooks
    "latex2e"
    "llncs"
    "llncs10"
    "hyperref"
    "graphicx"
    "comment"
    "amsmath"
    "amssymb"
    "color")
   (TeX-add-symbols
    '("refapp" 1)
    '("reftab" 1)
    '("refsec" 1)
    '("reffig" 1)
    "ECCVSubNumber")
   (LaTeX-add-labels
    "introduction"
    "related_work"
    "methodology"
    "results"
    "conclusion")
   (LaTeX-add-bibliographies
    "egbib"))
 :latex)

