(TeX-add-style-hook
 "p-presi"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("beamer" "18pt")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("adjustbox" "export")))
   (add-to-list 'LaTeX-verbatim-environments-local "semiverbatim")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "beamer"
    "beamer10"
    "templates/beamerthemekit"
    "adjustbox"
    "tikz"
    "url"
    "amsmath"
    "marvosym"
    "stmaryrd"
    "textcomp"
    "svg")
   (TeX-add-symbols
    '("semitransp" ["argument"] 1)))
 :latex)

