// MathJax configuration for MindFractal Lab
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    macros: {
      // Number sets
      R: "\\mathbb{R}",
      C: "\\mathbb{C}",
      Z: "\\mathbb{Z}",
      N: "\\mathbb{N}",

      // Common spaces
      Pcal: "\\mathcal{P}",
      Bcal: "\\mathcal{B}",
      Acal: "\\mathcal{A}",
      Fcal: "\\mathcal{F}",

      // Vectors and matrices
      vx: "\\mathbf{x}",
      vz: "\\mathbf{z}",
      vc: "\\mathbf{c}",
      mA: "\\mathbf{A}",
      mB: "\\mathbf{B}",
      mW: "\\mathbf{W}",
      mU: "\\mathbf{U}",
      mI: "\\mathbf{I}",
      mJ: "\\mathbf{J}",

      // Operators
      diag: "\\operatorname{diag}",
      sech: "\\operatorname{sech}",
      orbit: "\\operatorname{orbit}",

      // Norms
      norm: ["\\left\\| #1 \\right\\|", 1],
      abs: ["\\left| #1 \\right|", 1],

      // Lyapunov
      lyap: "\\lambda"
    }
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

document$.subscribe(() => {
  MathJax.typesetPromise()
})
