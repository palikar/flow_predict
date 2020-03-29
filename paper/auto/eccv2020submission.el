(TeX-add-style-hook
 "eccv2020submission"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("appendix" "title") ("hyperref" "pagebackref")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "href")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "llncs"
    "llncs10"
    "appendix"
    "hyperref"
    "subcaption"
    "graphicx"
    "comment"
    "amsmath"
    "amssymb"
    "color"
    "float")
   (TeX-add-symbols
    '("refapp" 1)
    '("reftab" 1)
    '("refsec" 1)
    '("reffig" 1)
    "thickhline"
    "ECCVSubNumber")
   (LaTeX-add-labels
    "introduction"
    "related_work"
    "methodology"
    "eval"
    "fig:single_images"
    "fig:single_psnr"
    "tab:single"
    "fig:rec_const_psnr"
    "fig:rec_speed_psnr"
    "fig:rec_fluid_psnr"
    "tab:recursive_inflow"
    "tab:recursive_fluid"
    "conclusion"
    "fig:const_sim"
    "fig:fluid_sim"
    "app1")
   (LaTeX-add-bibliographies
    "egbib"))
 :latex)

