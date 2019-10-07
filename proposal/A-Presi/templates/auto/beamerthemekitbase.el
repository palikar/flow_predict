(TeX-add-style-hook
 "beamerthemekitbase"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("babel" "english") ("textpos" "absolute" "overlay") ("overpic" "abs") ("helvet" "scaled=.92")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "babel"
    "ifthen"
    "hyperref"
    "textpos"
    "templates/semirounded"
    "overpic"
    "helvet"
    "templates/beamercolorthemekit")
   (TeX-add-symbols
    "beginbackup"
    "backupend"
    "titleimage"
    "titlelogo")
   (LaTeX-add-counters
    "framenumbervorappendix"))
 :latex)

