"""
Builds a Markdown report for the Lagrangian duality notebook.
"""

from IPython.display import Markdown, display

class Reporter:
    def __init__(self, title, a, u):
        self.title = title
        self.a = a
        self.u = u
    
    def results(self, x, p, lam) :
        self.x = x
        self.p = p
        self.lam = lam

    def results_p(self, x_p, p_p):
        self.x_p = x_p
        self.p_p = p_p

    def output(self, level=1):
        
        header = "#" * level
        rpt = f"""
{header} {self.title}

{header}# Solving with
- $a = {self.a}$
- $u = {self.u}$
{header}# Unperturbed Problem
- $x^* = {self.x}$
- $p^* = {self.p}$
- $λ^* = {self.lam}$
{header}# Perturbed Problem
- $x^* = {self.x_p}$
- $p^* = {self.p_p}$
{header}# Ratio
$$\\frac{{p^*(u)-p^*(0)}}{{u}} = {(self.p_p - self.p)/self.u}$$
"""
        display(Markdown(rpt))