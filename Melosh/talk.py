from manim_slide import *
import numpy as np
from pylab import cm
import matplotlib

# config.background_color = "#161c20"

toc = Group(
    Tex("1. Constitutive modeling"),
    Tex("2. Polyconvexity"),
    Tex("3. Neural ODE"),
    Tex("4. Results"),
    Tex("5. Application to FEM"),
    Tex("6. Conclusions")
).arrange(DOWN,aligned_edge=LEFT,buff=0.4)

class Title(SlideScene):
    def construct(self):
        title = Tex(r'Automatically polyconvex strain energy functions \\ with Neural ODEs').scale(1.25).shift(2.5*UP)
        name = Tex(r'Vahidullah Tac')
        collab = Tex('A. Buganza Tepole, F. Sahli Costabal').scale(0.8).next_to(name,DOWN).shift(0.3*DOWN)
        purdue=SVGMobject("purdue_logo.svg").shift(2.5*DOWN).scale(1/3)#.next_to(1.5*DOWN,LEFT,buff=2.5)

        self.play(FadeIn(name))
        self.slide_break()
        self.play(FadeIn(title))
        self.slide_break()
        self.play(FadeIn(collab))
        self.play(Write(purdue))
        self.slide_break()

        self.play(FadeOut(name, title, purdue, collab))
        self.slide_break()

        self.play(FadeIn(toc))
        self.slide_break()

        self.play(toc[0].animate.scale(1.2).set_color(YELLOW))
        self.slide_break()

        for i in range(1, len(toc)):
            self.play(toc[i].animate.scale(1.2).set_color(YELLOW),toc[i-1].animate.scale(1/1.2).set_color(WHITE))
            self.slide_break()

        self.play(toc[-1].animate.scale(1/1.2).set_color(WHITE))

class Intro(SlideScene):
    def construct(self):
        self.add(toc)

        heading = toc[0].copy().move_to(ORIGIN).scale(1.25).to_corner(UP)
        self.play(FadeOut(toc), ReplacementTransform(toc[0],heading))
        self.slide_break()

        expertmodels = Tex("Expert-constructed constitutive models").move_to(2*UP)
        limitations = BulletedList("Requires expert knowledge", "Poor fitting").set_color(RED).move_to(0.72*UP)
        ddmodels = Tex("$\\rightarrow$", " data-driven strain energy functions").move_to(DOWN)
        PImodels = Tex("Physics-informed", " data-driven strain energy functions").move_to(2*UP)
        PImodels[0].set_color(YELLOW)
        self.play(Write(expertmodels))
        self.slide_break()
        self.play(FadeIn(limitations[0],shift=0.5*UP))
        self.slide_break()
        self.play(FadeIn(limitations[1],shift=0.5*UP))
        self.slide_break()
        self.play(Write(ddmodels))

        self.play(FadeOut(expertmodels, limitations))
        self.slide_break()
        self.play(ReplacementTransform(ddmodels, PImodels))

        consistent = Tex("Consistent", " data-driven strain energy functions")
        objective = Tex("Objective", " data-driven strain energy functions")
        polyconvex = Tex("Polyconvex", " data-driven strain energy functions")
        
        for obj in [consistent, objective, polyconvex]:
            obj.move_to(2*UP)
            obj[0].set_color(YELLOW)
        
        self.slide_break()
        self.play(ReplacementTransform(PImodels, consistent))
        self.wait()
        self.play(ReplacementTransform(consistent, objective))
        self.wait()
        self.play(ReplacementTransform(objective, polyconvex))
        self.slide_break()
        
        benefits = Group(
            Tex("$\\bullet$ Physically reasonable"),
            Tex("$\\bullet$ Numerically stable ", "$\\rightarrow$ Important for FEM")
        ).arrange(DOWN,aligned_edge=LEFT).align_to(polyconvex, LEFT+UP).shift(DOWN+0.3*RIGHT)
        self.play(Write(benefits[0]))
        self.slide_break()
        self.play(Write(benefits[1][0]))
        self.slide_break()
        self.play(Write(benefits[1][1]))
        self.slide_break()
        self.play(FadeOut(benefits, polyconvex))

        NR = Tex("Newton-Raphson iterations").move_to(2.2*UP)
        self.play(Write(NR))
        self.slide_break()

        fun1  = lambda x: 0.2*np.exp(x)-0.25
        dfun1 = lambda x: 0.2*np.exp(x)
        fun2  = lambda x: (1.2*x-0.5)**3-1.5*(1.2*x-0.5)**2+0.7
        dfun2 = lambda x: 3*1.2*(1.2*x-0.5)**2 - 2*1.5*1.2*(1.2*x-0.5)
        axis1 = Axes([0,2.2], [0,1], x_length=8, axis_config={"include_ticks":False}).scale(0.5).shift(DOWN)
        xlabel = MathTex("x").move_to(axis1.coords_to_point(2.35,0))
        flabel = MathTex("\mathbf{f}(x)").move_to(axis1.coords_to_point(0,1.15))
        title = Tex("Convex").move_to(axis1.coords_to_point(1,1.25)).set_color(YELLOW)
        x = np.linspace(-0.5,1.8,20)
        y = fun1(x)
        graph = axis1.plot_line_graph(x, y, add_vertex_dots=False)
        self.play(Create(axis1))
        self.play(FadeIn(xlabel, flabel, title))
        self.slide_break()
        self.play(Create(graph))
        self.slide_break()

        
        xval = 1.7
        x0label = MathTex("x_0").move_to(axis1.coords_to_point(xval, -0.2))
        yprev = None
        for i in range(4):
            x = Dot(axis1.coords_to_point(xval,0))
            y = Dot(axis1.coords_to_point(xval, fun1(xval)))
            line1 = Line(x,y)
            xval = xval - fun1(xval)/dfun1(xval)
            line2 = Line(y, axis1.coords_to_point(xval,0))
            self.play(FadeIn(x0label), Create(x), run_time=0.6)
            self.play(FadeOut(yprev), run_time=0.3)
            self.play(ShowPassingFlash(line1), run_time=0.6)
            self.play(Create(y), run_time=0.3)
            self.play(FadeOut(x), FadeOut(x0label), run_time=0.3)
            self.play(ShowPassingFlash(line2), run_time=0.6)
            yprev = y
            x0label = None

        self.wait()
        self.slide_break()
        self.play(*[obj.animate.shift(3*LEFT) for obj in [axis1, graph, xlabel, flabel, title, yprev]])

        axis2 = Axes([0,2.2], [0,1], x_length=8, axis_config={"include_ticks":False}).scale(0.5).shift(DOWN+3*RIGHT)
        xlabel2 = MathTex("x").move_to(axis2.coords_to_point(2.35,0))
        flabel2 = MathTex("\mathbf{g}(x)").move_to(axis2.coords_to_point(0,1.15))
        title2 = Tex("Non-convex").move_to(axis2.coords_to_point(1.2,1.25)).set_color(YELLOW)
        x = np.linspace(-0.5,1.8,20)
        y = fun2(x)
        graph2 = axis2.plot_line_graph(x, y, add_vertex_dots=False)
        self.play(Create(axis2))
        self.play(FadeIn(xlabel2, flabel2, title2))
        self.play(Create(graph2))
        self.slide_break()

        xval = 1.7
        x0label = MathTex("x_0").move_to(axis2.coords_to_point(xval, -0.2))
        yprev2 = None
        for i in range(6):
            x = Dot(axis2.coords_to_point(xval,0))
            y = Dot(axis2.coords_to_point(xval, fun2(xval)))
            line1 = Line(x,y)
            xval = xval - fun2(xval)/dfun2(xval)
            line2 = Line(y, axis2.coords_to_point(xval,0))
            self.play(FadeIn(x0label), Create(x), run_time=0.6)
            self.play(FadeOut(yprev2), run_time=0.3)
            self.play(ShowPassingFlash(line1), run_time=0.6)
            self.play(Create(y), run_time=0.3)
            self.play(FadeOut(x), FadeOut(x0label), run_time=0.3)
            self.play(ShowPassingFlash(line2), run_time=0.6)
            yprev2 = y
            x0label = None
        
        self.slide_break()
        self.play(FadeOut(axis1, graph, xlabel, flabel, title, yprev, axis2, graph2, xlabel2, flabel2, title2, yprev2, NR))
        self.slide_break()
        self.play(FadeIn(polyconvex))

        workarounds = Group(
            Tex(r"$\bullet$ Ignored"),
            Tex(r"$\bullet$ Convexity in $\mathbf{C}$?"),
            Tex(r"$\bullet$ Numerical approximations")
        ).arrange(DOWN,aligned_edge=LEFT).align_to(polyconvex, LEFT+UP).shift(DOWN+0.3*RIGHT)
        unfit = Tex(r"$\rightarrow$ Unfit for FEM").align_to(workarounds, LEFT+DOWN).shift(DOWN+0.3*LEFT)
        self.slide_break()
        self.play(Write(workarounds[0]))
        self.slide_break()
        self.play(Write(workarounds[1]))
        self.slide_break()
        self.play(Write(workarounds[2]))
        self.slide_break()
        self.play(Write(unfit))
        self.slide_break()
        self.play(FadeOut(workarounds, polyconvex, unfit))

        canwepoly = Tex("Can we guarantee ", "polyconvexity", " in a data-driven framework?")
        canwepoly[1].set_color(YELLOW)
        self.play(FadeIn(canwepoly))
        self.slide_break()
        self.play(FadeOut(canwepoly, heading))

class Polyconvexity(SlideScene):
    def construct(self):
        self.play(FadeIn(toc))
        self.slide_break()

        heading = toc[1].copy().move_to(ORIGIN).scale(1.25).to_corner(UP)
        self.play(FadeOut(toc), ReplacementTransform(toc[1],heading))
        self.slide_break()

        whatispoly = Tex("What is ","polyconvexity","?")
        whatispoly[1].set_color(YELLOW)
        self.play(Write(whatispoly))
        self.slide_break()
        self.play(whatispoly.animate.move_to(2.2*UP))

        Psi = MathTex("\Psi", " = \hat{\Psi}(", "\mathbf{F}", ", ", "\mathrm{cof}\mathbf{F}", ", ", "\det\mathbf{F}", ")")
        self.play(Write(Psi.get_part_by_tex("\Psi")), Write(Psi.get_part_by_tex(" = \hat{\Psi}(")), Write(Psi.get_part_by_tex(")")))
        self.slide_break()
        self.play(Write(Psi.get_part_by_tex("\mathbf{F}")))
        self.play(FadeIn(Psi[3]))
        self.play(Write(Psi.get_part_by_tex("\mathrm{cof}\mathbf{F}")))
        self.play(FadeIn(Psi[5]))
        self.play(Write(Psi.get_part_by_tex("\det\mathbf{F}")))
        self.slide_break()

        Psi2 = MathTex("\Psi", " = \Psi_{\mathbf{F}}(", "\mathbf{F}",
         ") + \Psi_{\mathrm{cof}\mathbf{F}}(", "\mathrm{cof}\mathbf{F}", ") + \Psi_{\det\mathbf{F}}(", "\det\mathbf{F}", ")")
        self.play(Transform(Psi[0],Psi2[0]), Transform(Psi[1:], Psi2[1:]), FadeOut(whatispoly))
        self.slide_break()

        self.play(Circumscribe(Psi2[2]), Circumscribe(Psi2[4]), Circumscribe(Psi2[6]))
        self.slide_break()

        self.remove(Psi)
        Psi3 = MathTex("\Psi", " = \Psi_{I_1}(", "I_1", ") + \Psi_{I_2}(", "I_2", ") + \Psi_{I_3}(", "I_3", ") + \Psi_{I_{4v}}(", "I_{4v}", ") + ...")
        self.play(ReplacementTransform(Psi[0],Psi3[0]), ReplacementTransform(Psi[1:], Psi3[1:]))
        self.slide_break()
        
        self.remove(Psi2)
        self.play(Psi3.animate.move_to(2.2*UP))

        invs = Group(
            MathTex("I_1 = \hat{I}_1(\mathbf{F})"),
            MathTex("I_2 = \hat{I}_2(cof\mathbf{F})"),
            MathTex("I_3 = \hat{I}_3(\det\mathbf{F})"),
            MathTex("I_{4v} = \hat{I}_{4v}(\mathbf{F})"),
        )
        self.play(FadeIn(invs[0], shift=UP))
        self.play(FadeOut(invs[0], shift=UP), FadeIn(invs[1], shift=UP))
        self.play(FadeOut(invs[1], shift=UP), FadeIn(invs[2], shift=UP))
        self.play(FadeOut(invs[2], shift=UP), FadeIn(invs[3], shift=UP))
        self.play(FadeOut(invs[-1], shift=UP))
        self.slide_break()

        Psi3cont = MathTex("+ \Psi_{I_1+I_2}(", "I_1 + I_2", ") + \Psi_{I_1 + I_{4v}}(", "I_1 + I_{4v}", ") + ...").next_to(Psi3, DOWN).shift(RIGHT)
        self.play(FadeIn(Psi3cont))
        self.slide_break()

        self.play(Circumscribe(Psi3[0]))
        self.slide_break()

        self.play(FadeOut(Psi3), FadeOut(Psi3cont))

        egeqn = MathTex("f(x)", " = ", "g", "(", "h", "(x))")
        self.play(Write(egeqn))
        self.slide_break()
        self.play(egeqn.animate.move_to(2.2*UP))

        h = egeqn[4].copy()
        self.play(h.animate.move_to(ORIGIN).move_to(5*LEFT+UP))
        convex = Tex("Convex").next_to(h,DOWN).set_color(GREEN)
        nondec = Tex("Non-decreasing").next_to(convex,DOWN).set_color(BLUE)
        axes = Axes([-1,1], [-1,1], x_length=6).scale(0.4).next_to(nondec, DOWN)
        x = np.linspace(-1,1,20)
        y = x**2
        graph = axes.plot_line_graph(x, y, add_vertex_dots=False)
        self.play(FadeIn(convex), Create(axes))
        self.play(Create(graph))

        g = egeqn[2].copy()
        self.play(g.animate.move_to(ORIGIN).move_to(UP))
        convex2 = Tex("Convex").next_to(g,DOWN).set_color(GREEN)
        nondec2 = Tex("Non-decreasing").next_to(convex2,DOWN).set_color(BLUE)
        axes2 = Axes([-1,1], [-1,1], x_length=6).scale(0.4).next_to(nondec2, DOWN)
        graph2 = axes2.plot_line_graph(x, y, add_vertex_dots=False)
        graph2_2 = axes2.plot_line_graph(x, 0.1*10**x, add_vertex_dots=False)
        self.play(FadeIn(convex2), Create(axes2))
        self.play(Create(graph2))
        self.slide_break()
        self.play(FadeIn(nondec2))
        self.play(Transform(graph2, graph2_2))
        self.slide_break()

        fconv = Tex("$\implies f(x)$ is convex").move_to(RIGHT*4.5+UP)
        self.play(Write(fconv))
        self.slide_break()

        self.remove(egeqn, h, convex, nondec, axes, g, convex2, nondec2, axes2)
        self.play(*[FadeOut(item) for item in [egeqn, h, convex, axes, graph, g, convex2, nondec2, axes2, graph2, graph2_2, fconv]])
        
        Psi = Group(
            MathTex("\Psi", " = ", "\Psi_{I_1}", "(", "I_1", ") + ", "\Psi_{I_2}", "(", "I_2", ") + ", "\Psi_{I_3}", 
                        "(", "I_3", ") + ", "\Psi_{I_{4v}}", "(", "I_{4v}", ") + ..."),
            MathTex("+ ", "\Psi_{I_1+I_2}", "(", "I_1 + I_2", ") + ", "\Psi_{I_1 + I_{4v}}", 
                        "(", "I_1 + I_{4v}", ") + ...")
        ).arrange(DOWN).move_to(UP*2)
        self.play(FadeIn(Psi))


        self.play(*[Circumscribe(Psi[0][i], run_time=1.7) for i in [4, 8, 12, 16]], *[Circumscribe(Psi[1][i], run_time=1.7) for i in [3, 7]])
        self.slide_break()
        self.play(*[Circumscribe(Psi[0][i], run_time=1.7) for i in [2, 6, 10, 14]], *[Circumscribe(Psi[1][i], run_time=1.7) for i in [1, 5]])
        self.slide_break()

        convtext = Group(
            Tex("$I_1, I_2, I_3, I_{4v}... \\rightarrow$", " Convex \checkmark"),
            Tex("$\Psi_{I_1}, \Psi_{I_2}, \Psi_{I_3}... \\rightarrow$", " Convex \& non-decreasing?")
        ).arrange(DOWN,aligned_edge=LEFT,buff=0.4).move_to(DOWN*0.5)

        convtext[0][1].set_color(YELLOW)

        self.play(Write(convtext[0]))
        self.slide_break()
        self.play(Write(convtext[1]))
        self.slide_break()
        
        nodetext = Tex("$\Psi_{I_1}, \Psi_{I_2}, \Psi_{I_3}... \\rightarrow$", " Neural ODEs").align_to(convtext[1],LEFT+DOWN)
        nodetext[1].set_color(YELLOW)
        self.play(Transform(convtext[1][1], nodetext[1]))
        self.slide_break()

        self.play(FadeOut(convtext), FadeOut(nodetext))
        self.slide_break()

        self.play(Flash(Psi[0][0]))
        self.slide_break()

        dPsi = Group(
            MathTex("\\frac{\partial \Psi}{\partial \mathbf{C}}", "\\rightarrow", "\mathbf{S}"),
            MathTex("\\frac{\partial^2 \Psi}{\partial \mathbf{C} \partial \mathbf{C}}", "\\rightarrow", "\mathbb{C}")
        ).arrange(DOWN, buff=0.6).move_to(DOWN)

        self.play(FadeIn(dPsi[0][0]), FadeIn(dPsi[1][0]))
        self.slide_break()
        self.play(FadeIn(dPsi[0][1:]), FadeIn(dPsi[1][1:]))
        self.slide_break()
        self.play(Circumscribe(dPsi))
        self.slide_break()
        self.play(FadeOut(dPsi))
        self.slide_break()

        self.play(Psi.animate.move_to(DOWN*0.4))


        Psi2 = Group(
            MathTex("\Psi", " = ", "\Psi_{I_1}(I_1)", " + ", "\Psi_{I_2}(I_2)", " + ", "\Psi_{I_3}(I_3)", " + ", 
                    "\Psi_{I_{4v}}(I_{4v})", " + ..."),
            MathTex("+ ", "\Psi_{I_1+I_2}(I_1 + I_2)", " + ", "\Psi_{I_1 + I_{4v}}(I_1 + I_{4v})", " + ...")
        ).arrange(DOWN).move_to(UP*2).move_to(DOWN*0.4)
        self.play(FadeIn(Psi2))
        self.add(Psi2)
        self.remove(Psi)

        dPsi1 = MathTex("\\frac{\partial \Psi}{\partial \mathbf{C}}", " = ", "\Psi_{I_1}(I_1)", " + ", "\Psi_{I_2}(I_2)", " + ", "\Psi_{I_3}(I_3)", " + ", 
                    "\Psi_{I_{4v}}(I_{4v})", " + ...").align_to(Psi2[0], LEFT+DOWN)
        dPsi2 = MathTex("\\frac{\partial \Psi}{\partial \mathbf{C}}", " = ", 
                        "\\frac{\partial \Psi_{I_1}}{\partial I_1}\\frac{\partial I_1}{\partial \mathbf{C}}", 
                        " + ", "\Psi_{I_2}(I_2)", " + ", "\Psi_{I_3}(I_3)", " + ", 
                        "\Psi_{I_{4v}}(I_{4v})", " + ...").align_to(Psi2[0], LEFT+DOWN)
        dPsi3 = MathTex("\\frac{\partial \Psi}{\partial \mathbf{C}}", " = ", 
                        "\\frac{\partial \Psi_{I_1}}{\partial I_1}\\frac{\partial I_1}{\partial \mathbf{C}}", 
                        " + ", "\\frac{\partial \Psi_{I_2}}{\partial I_2}\\frac{\partial I_2}{\partial \mathbf{C}}", 
                        " + ", "\Psi_{I_3}(I_3)", " + ", 
                        "\Psi_{I_{4v}}(I_{4v})", " + ...").align_to(Psi2[0], LEFT+DOWN)
        dPsi4 = MathTex("\\frac{\partial \Psi}{\partial \mathbf{C}}", " = ", 
                        "\\frac{\partial \Psi_{I_1}}{\partial I_1}\\frac{\partial I_1}{\partial \mathbf{C}}", 
                        " + ", "\\frac{\partial \Psi_{I_2}}{\partial I_2}\\frac{\partial I_2}{\partial \mathbf{C}}", 
                        " + ", "\\frac{\partial \Psi_{I_3}}{\partial I_3}\\frac{\partial I_3}{\partial \mathbf{C}}", " + ", 
                        "\Psi_{I_{4v}}(I_{4v})", " + ...").align_to(Psi2[0], LEFT+DOWN)
        dPsi5 = MathTex("\\frac{\partial \Psi}{\partial \mathbf{C}}", " = ", 
                        "\\frac{\partial \Psi_{I_1}}{\partial I_1}\\frac{\partial I_1}{\partial \mathbf{C}}", 
                        " + ", "\\frac{\partial \Psi_{I_2}}{\partial I_2}\\frac{\partial I_2}{\partial \mathbf{C}}", 
                        " + ", "\\frac{\partial \Psi_{I_3}}{\partial I_3}\\frac{\partial I_3}{\partial \mathbf{C}}", " + ...").align_to(Psi2[0], LEFT+DOWN)

        self.play(ReplacementTransform(Psi2[0], dPsi1))
        self.add(dPsi1)
        self.remove(*Psi2[0])
        self.play(ReplacementTransform(dPsi1, dPsi2))
        self.add(dPsi2)
        self.remove(dPsi1)
        self.play(ReplacementTransform(dPsi2, dPsi3))
        self.add(dPsi3)
        self.remove(dPsi2)
        self.play(ReplacementTransform(dPsi3, dPsi4))
        self.add(dPsi4)
        self.remove(dPsi3)
        self.play(ReplacementTransform(dPsi4[7:], dPsi5[7]), FadeOut(Psi2[1]))
        self.add(dPsi5)
        self.remove(dPsi4)
        self.remove(Psi2[1])
        self.slide_break()
        
        self.remove(*dPsi1, *dPsi2, *dPsi3, *dPsi4, *Psi2[0], *Psi2[1])
        self.play(dPsi5.animate.move_to(UP*2.1))
        self.slide_break()

        convtext1 = Tex("$\Psi_{I_1}, \Psi_{I_2}, \Psi_{I_3}...$", " $\\rightarrow$ ", "Convex", " \& ", "Non-decreasing")
        convtext2 = Group(
            Tex("$\Psi_{I_i}$"),
            Tex("Convex"),
            Tex("Non-decreasing")
        ).arrange(DOWN, buff=0.5).move_to(LEFT*3+DOWN)
        convtext2[1].set_color(GREEN)
        convtext2[2].set_color(BLUE)
        self.play(FadeIn(convtext1))
        self.slide_break()
        self.play(Transform(convtext1[0], convtext2[0]), 
                    convtext1[2].animate.move_to(convtext2[1]),
                    convtext1[4].animate.move_to(convtext2[2]),
                    FadeOut(convtext1[1]),
                    FadeOut(convtext1[3]))
        self.add(convtext2)
        self.remove(*convtext1)

        convtext3 = Group(
            Tex("$\\frac{\partial \Psi_{I_i}}{\partial I_i}$"),
            Tex("?"),
            Tex("?")
        ).arrange(DOWN, buff=0.5).move_to(RIGHT*3).align_to(convtext2[2],DOWN)
        convtext3[1].set_color(GREEN)
        convtext3[2].set_color(BLUE)
        convtext4 = Group(
            Tex("$\\frac{\partial \Psi_{I_i}}{\partial I_i}$"),
            Tex("Monotonic"),
            Tex("Non-negative")
        ).arrange(DOWN, buff=0.5).move_to(RIGHT*3).align_to(convtext2[2],DOWN)
        convtext4[1].set_color(GREEN)
        convtext4[2].set_color(BLUE)
        self.play(*[Write(convtext3[i]) for i in range(3)])
        self.slide_break()

        self.play(Transform(convtext3[1], convtext4[1]))
        self.slide_break()
        self.play(Transform(convtext3[2], convtext4[2]))
        self.slide_break()

        self.remove(*convtext1)
        self.play(FadeOut(*convtext2))

        self.play(convtext3.animate.move_to(ORIGIN+DOWN*0.9))
        self.slide_break()
        self.play(FadeOut(*convtext3, dPsi5, heading))

class NODE(SlideMovingCameraScene):
    def construct(self):
        self.play(FadeIn(toc))
        self.slide_break()

        heading = toc[2].copy().move_to(ORIGIN).scale(1.25).to_corner(UP)
        self.play(FadeOut(toc), ReplacementTransform(toc[2],heading))
        self.slide_break()

        node1 = Tex("Neural ODE").set_color(YELLOW)
        node2 = MathTex("\mathcal{N}").set_color(YELLOW).scale(2)

        self.play(Write(node1))
        self.play(Transform(node1, node2))

        nodebox = Square(side_length=2.0)
        inparr = Arrow([-2.5,0,0], [-0.8,0,0])
        outarr = Arrow([+0.8,0,0], [+2.5,0,0])
        inptext = Tex("Input").move_to([-3.3,0,0])
        outtext = Tex("Output").move_to([+3.3,0,0])
        self.play(Create(nodebox))
        self.slide_break()
        self.play(FadeIn(inptext))
        self.play(Create(inparr))
        self.slide_break()
        self.play(Create(outarr))
        self.play(FadeIn(outtext))
        self.slide_break()

        self.camera.frame.save_state()
        self.play(FadeOut(node1), self.camera.frame.animate.set(width=1.8), run_time=2.5)
        self.remove(heading, node1, nodebox, inptext, outtext, inparr, outarr)
        self.play(Restore(self.camera.frame), run_time=0.2)

        ode = MathTex("\\frac{d\mathbf{h}}{dt}", "=", "f(\mathbf{h}(t),t,\\theta)").move_to(UP)
        inp = Tex("Input ", "$\\rightarrow$", " $\mathbf{h}(0)$").next_to(ode,DOWN)
        out = Tex("$\mathbf{h}(1)$ ", "$\\rightarrow$", " Output").next_to(inp,DOWN)
        self.play(FadeIn(ode))
        self.slide_break()
        self.play(Circumscribe(ode[2]))
        self.slide_break()
        self.play(FadeIn(inp, shift=UP))
        self.slide_break()
        self.play(FadeIn(out, shift=UP))
        self.slide_break()

        self.play(ode.animate.move_to(UP*3), FadeOut(inp), FadeOut(out))

        axis1 = Axes([0,1.25], [0,1.25], x_length=8, axis_config={"include_ticks":False}).scale(0.5)
        dashedline = DashedLine(axis1.coords_to_point(1,0), axis1.coords_to_point(1,1.15), dash_length=0.3)
        t = MathTex("t").move_to(axis1.coords_to_point(1.4,0))
        h = MathTex("\mathbf{h}(t)").move_to(axis1.coords_to_point(0,1.4))
        h0 = MathTex("\mathbf{h}(0)").move_to(axis1.coords_to_point(0,-0.2))
        h1 = MathTex("\mathbf{h}(1)").move_to(axis1.coords_to_point(1,-0.2))
        inp = Tex("(input)").scale(0.8).next_to(h0,DOWN)
        out = Tex("(output)").scale(0.8).next_to(h1,DOWN)
        self.play(Create(axis1))
        self.play(FadeIn(h))
        self.play(FadeIn(t))
        self.play(Write(h0))
        self.play(FadeIn(inp))
        self.play(Write(h1))
        self.play(Create(dashedline))
        self.play(FadeIn(out))
        self.slide_break()

        graphs = []
        inpdots = []
        outdots = []
        x = np.linspace(0,1)
        for i in range(5):
            y = np.exp(x+i*0.3)/10
            dot1 = Dot(axis1.coords_to_point(x[0],y[0]))
            dot2 = Dot(axis1.coords_to_point(x[-1],y[-1]))
            graph = axis1.plot_line_graph(x, y, add_vertex_dots=False)
            graphs.append(graph)
            inpdots.append(dot1)
            outdots.append(dot2)
            self.play(Create(dot1))
            self.play(Create(graph))
            self.play(Create(dot2))
            # self.slide_break()

        #Incorrect graph (that intersects)
        y = np.exp(x+5*0.3)/10 - x**5
        dot1 = Dot(axis1.coords_to_point(x[0],y[0]))
        graph = axis1.plot_line_graph(x[:39], y[:39], add_vertex_dots=False)
        cross = Cross().move_to(axis1.coords_to_point(x[38], y[38])).scale(0.5)
        self.play(Create(dot1))
        self.play(Create(graph), run_time=2 )
        self.play(Create(cross))
        
        self.remove(dot1, graph, cross)

        y = np.exp(x+5*0.3)/10
        dot1 = Dot(axis1.coords_to_point(x[0],y[0]))
        dot2 = Dot(axis1.coords_to_point(x[-1],y[-1]))
        graph = axis1.plot_line_graph(x, y, add_vertex_dots=False)
        graphs.append(graph)
        inpdots.append(dot1)
        outdots.append(dot2)
        self.play(Create(dot1))
        self.play(Create(graph))
        self.play(Create(dot2))
        self.slide_break()

        #Shift the entire axis with everything in it to the right
        self.remove(*graphs, *inpdots, *outdots)
        self.play(*[item.animate.shift(LEFT*2.8) for item in [axis1, t, h, h0, h1, dashedline, inp, out]])

        #Input-output map
        axis2 = Axes([0,0.6], [0.1,1.25], x_length=8, axis_config={"include_ticks":False}).scale(0.5).shift(RIGHT*2.8)
        inp2 = Tex("Input").move_to(axis2.coords_to_point(0.6,-0.12))
        out2 = Tex("Output").move_to(axis2.coords_to_point(0,1.4))
        self.play(Create(axis2))
        self.play(FadeIn(inp2))
        self.play(FadeIn(out2))

        #Plot everything again but this time simultaneously with the input-output map.
        graphs = []
        dots = []
        x = np.linspace(0,1)
        y = np.exp(x)/10
        dot1 = Dot(axis1.coords_to_point(x[0],y[0]))
        dot2 = Dot(axis1.coords_to_point(x[-1],y[-1]))
        graph = axis1.plot_line_graph(x, y, add_vertex_dots=False)
        dot = Dot(axis2.coords_to_point(y[0],y[-1]))
        endpoints1 = [y[0],y[-1]]
        graphs.append(graph)
        dots.append(dot1)
        dots.append(dot2)
        dots.append(dot)
        self.play(Create(dot1), run_time=0.5)
        self.play(Create(graph), run_time=0.5)
        self.play(Create(dot2), Create(dot), run_time=0.5)

        for i in range(1,6):
            y = np.exp(x+i*0.3)/10
            dot1 = Dot(axis1.coords_to_point(x[0],y[0]))
            dot2 = Dot(axis1.coords_to_point(x[-1],y[-1]))
            graph = axis1.plot_line_graph(x, y, add_vertex_dots=False)
            dot = Dot(axis2.coords_to_point(y[0],y[-1]))
            x2 = [endpoints1[0], y[0]]
            y2 = [endpoints1[1], y[-1]]
            graph2 = axis2.plot_line_graph(x2,y2, add_vertex_dots=False)
            graphs.append(graph)
            dots.append(dot1)
            dots.append(dot2)
            dots.append(dot1)
            self.play(Create(dot1), run_time=0.5)
            self.play(Create(graph), run_time=0.5)
            self.play(Create(dot), Create(dot2), run_time=0.5)
            self.play(Create(graph2), run_time=0.5)

        monotonic = Tex("Monotonic \checkmark").set_color(GREEN).move_to(RIGHT*4.9+UP*0.4)
        nonnegative = Tex("Non-negative \checkmark").set_color(BLUE).next_to(monotonic,DOWN)

        self.play(Write(monotonic))
        self.slide_break()

        #Plot the 0-0 line
        y = np.exp(x)/10
        endpoints1 = [y[0],y[-1]]
        y = x*0
        dot1 = Dot(axis1.coords_to_point(x[0],y[0]))
        dot2 = Dot(axis1.coords_to_point(x[-1],y[-1]))
        graph = axis1.plot_line_graph(x, y, add_vertex_dots=False)
        dot = Dot(axis2.coords_to_point(0,0.1))
        x2 = [endpoints1[0], 0]
        y2 = [endpoints1[1], 0.1]
        graph2 = axis2.plot_line_graph(x2,y2, add_vertex_dots=False)
        graphs.append(graph)
        dots.append(dot1)
        dots.append(dot2)
        dots.append(dot)
        self.play(Create(dot1))
        self.play(Flash(dot1))
        self.play(Create(graph))
        self.play(Create(dot2), Create(dot))
        self.play(Flash(dot2))
        self.play(Create(graph2), run_time=0.5)
        self.play(Flash(dot))
        self.play(Write(nonnegative))
        self.slide_break()


        self.play(*[FadeOut(obj) for obj in self.mobjects])

        self.camera.frame.save_state()
        self.play(self.camera.frame.animate.set(width=1.8), run_time=0.2)
        self.add(heading, nodebox, inptext, outtext, inparr, outarr)
        self.play(Restore(self.camera.frame), FadeIn(node1), run_time=1.5)
        self.slide_break()

        self.play(*[obj.animate.shift(UP*1.3) for obj in [node1, nodebox, inptext, outtext, inparr, outarr]])

        dPsi = MathTex("\\frac{\partial \Psi}{\partial \mathbf{C}}", " = ", 
                        "\\frac{\partial \Psi_{I_1}}{\partial I_1}", "\\frac{\partial I_1}{\partial \mathbf{C}}", 
                        " + ", "\\frac{\partial \Psi_{I_2}}{\partial I_2}", "\\frac{\partial I_2}{\partial \mathbf{C}}", 
                        " + ", "\\frac{\partial \Psi_{I_3}}{\partial I_3}", "\\frac{\partial I_3}{\partial \mathbf{C}}", " + ...").shift(DOWN*1.5)
        dPsi2 = MathTex("\\frac{\partial \Psi}{\partial \mathbf{C}}", " = ", 
                        "\mathcal{N}_1(I_1)", "\\frac{\partial I_1}{\partial \mathbf{C}}", 
                        " + ", "\\frac{\partial \Psi_{I_2}}{\partial I_2}", "\\frac{\partial I_2}{\partial \mathbf{C}}", 
                        " + ", "\\frac{\partial \Psi_{I_2}}{\partial I_2}", "\\frac{\partial I_3}{\partial \mathbf{C}}", " + ...").shift(DOWN*1.5)
        dPsi3 = MathTex("\\frac{\partial \Psi}{\partial \mathbf{C}}", " = ", 
                        "\mathcal{N}_1(I_1)", "\\frac{\partial I_1}{\partial \mathbf{C}}", 
                        " + ", "\mathcal{N}_2(I_2)", "\\frac{\partial I_2}{\partial \mathbf{C}}", 
                        " + ", "\\frac{\partial \Psi_{I_2}}{\partial I_2}", "\\frac{\partial I_3}{\partial \mathbf{C}}", " + ...").shift(DOWN*1.5)
        dPsi4 = MathTex("\\frac{\partial \Psi}{\partial \mathbf{C}}", " = ", 
                        "\mathcal{N}_1(I_1)", "\\frac{\partial I_1}{\partial \mathbf{C}}", 
                        " + ", "\mathcal{N}_2(I_2)", "\\frac{\partial I_2}{\partial \mathbf{C}}", 
                        " + ", "\mathcal{N}_3(I_3)", "\\frac{\partial I_3}{\partial \mathbf{C}}", " + ...").shift(DOWN*1.5)

        dPsi2[2].set_color(YELLOW)
        dPsi3[2].set_color(YELLOW)
        dPsi4[2].set_color(YELLOW)
        dPsi3[5].set_color(YELLOW)
        dPsi4[5].set_color(YELLOW)
        dPsi4[8].set_color(YELLOW)
        self.play(FadeIn(dPsi))
        self.slide_break()
        self.play(*[Circumscribe(dPsi[i]) for i in [2, 5, 8]])
        self.slide_break()
        self.play(Transform(dPsi, dPsi2))
        self.add(dPsi2)
        self.remove(dPsi)
        self.play(Transform(dPsi2, dPsi3))
        self.add(dPsi3)
        self.remove(dPsi2)
        self.play(Transform(dPsi3, dPsi4))
        self.add(dPsi4)
        self.remove(dPsi3)
        self.slide_break()

        self.play(FadeOut(node1, nodebox, inptext, outtext, inparr, outarr))
        self.play(dPsi4.animate.move_to(ORIGIN))
        self.remove(*[obj for obj in self.mobjects])
        self.slide_break()

class Results_p1(SlideScene):
    def construct(self):
        self.play(FadeIn(toc))
        self.slide_break()

        heading = toc[3].copy().move_to(ORIGIN).scale(1.25).to_corner(UP)
        self.play(FadeOut(toc), ReplacementTransform(toc[3],heading))
        self.slide_break()

        GOH_gt = SVGMobject("convexity_GOH_gt.svg").scale(2.5).shift(DOWN+0.6*LEFT)
        GOH_pr = SVGMobject("convexity_GOH_pr.svg").scale(2.5).move_to(GOH_gt).shift(3.5*RIGHT)
        Fung_gt = SVGMobject("convexity_Fung_gt.svg").scale(2.5)
        Fung_pr = SVGMobject("convexity_Fung_pr.svg").scale(2.5)
        txt1 = SVGMobject("convexity_text.svg").scale(2.5).move_to(GOH_gt)
        txt2 = txt1.copy().move_to(GOH_pr)
        label1 = Tex("GOH data").next_to(GOH_gt, UP).shift(0.6*RIGHT)
        label2 = Tex("N-ODE predictions").next_to(GOH_pr, UP).shift(0.6*RIGHT)
        self.play(FadeIn(label1))
        self.slide_break()
        self.play(Write(GOH_gt), Write(txt1), run_time=3)
        self.slide_break()
        self.play(GOH_gt.animate.shift(3*LEFT), txt1.animate.shift(3*LEFT), label1.animate.shift(3*LEFT))
        self.play(Write(GOH_pr), Write(txt2), FadeIn(label2))
        self.slide_break()


        conv_shadow = SVGMobject("convexity_shadow.svg").scale(2.5).move_to(GOH_gt)
        train_reg = Tex("Training region").scale(0.5).move_to(GOH_gt).shift(1.5*RIGHT+1.5*UP).set_color(BLACK)
        self.play(FadeIn(conv_shadow))
        self.play(Write(train_reg))
        self.slide_break()
        self.play(FadeOut(conv_shadow), FadeOut(train_reg))
        self.slide_break()

        conv_outline1 = SVGMobject("convexity_outline1.svg").scale(2.5).move_to(GOH_pr)
        conv_outline2 = SVGMobject("convexity_outline2.svg").scale(2.5).move_to(GOH_pr)
        conv_outline3 = SVGMobject("convexity_outline3.svg").scale(2.5).move_to(GOH_pr)
        self.play(ShowPassingFlash(conv_outline1), run_time=2)
        self.play(ShowPassingFlash(conv_outline2), run_time=3)
        self.play(ShowPassingFlash(conv_outline3), run_time=3)
        self.slide_break()

        conv_outline1.move_to(GOH_gt)
        conv_outline2.move_to(GOH_gt)
        conv_outline3.move_to(GOH_gt)
        self.play(ShowPassingFlash(conv_outline1), ShowPassingFlash(conv_outline2), ShowPassingFlash(conv_outline3), run_time=2)
        self.slide_break()


        self.play(*[FadeOut(obj) for obj in [GOH_gt, GOH_pr, txt1, txt2, label1, label2]])
        self.slide_break()

        Fung_gt.move_to(GOH_gt)
        Fung_pr.move_to(GOH_pr)
        label3 = Tex("Fung data").next_to(Fung_gt, UP).shift(0.6*RIGHT)
        label4 = label2.copy()
        self.play(Write(Fung_gt), Write(txt1), FadeIn(label3))
        self.slide_break()
        self.play(Write(Fung_pr), Write(txt2), FadeIn(label4))
        self.slide_break()

        cross = Cross().move_to(Fung_gt).shift(0.5*RIGHT+0.5*UP)
        self.play(Create(cross))
        self.slide_break()

        self.play(*[FadeOut(obj) for obj in [Fung_gt, Fung_pr, txt1, txt2, label3, label4, cross]])

class Results_p2(SlideThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=0, theta=0, gamma=90*DEGREES)
        heading = toc[3].copy().move_to(ORIGIN).scale(1.25).to_corner(UP)
        self.add(heading)
        # self.add(toc)
        # self.slide_break()

        # heading = toc[3].copy().move_to(ORIGIN).scale(1.25).to_corner(UP)
        cbar = ImageMobject('cbar.png').move_to(ORIGIN).move_to(6*LEFT).scale(2)
        cbar_label = Tex(r'Error \\\ [MPa]').move_to(ORIGIN).move_to(5.2*LEFT).scale(0.7)
        cbar_tick1 = Tex("0.0").align_to(cbar_label, LEFT).shift(1.7*DOWN).scale(0.7)
        cbar_tick2 = Tex("0.15").align_to(cbar_label, LEFT).shift(1.7*UP).scale(0.7)
        self.add_fixed_in_frame_mobjects(heading, cbar, cbar_label, cbar_tick1, cbar_tick2)
        self.remove(cbar, cbar_label, cbar_tick1, cbar_tick2)
        self.slide_break()

        # Plots
        x_range = [1.0, 1.17, 0.05]
        y_range = [  0,  1.5,  0.5]
        z_range = [1.0, 1.25, 0.05]
        axis_node = ThreeDAxes(
            x_range, 
            y_range, 
            z_range, 
            z_length=10.5, 
            axis_config={'include_numbers':True,
                         'font_size':96}
            ).scale(0.4).move_to(ORIGIN)
        x_axis = axis_node.get_x_axis()
        y_axis = axis_node.get_y_axis()
        z_axis = axis_node.get_z_axis()
        x_label = Tex("$\lambda_{x}$").next_to(x_axis, RIGHT)
        y_label = Tex("$\mathbf{\sigma}_{1}$", "[MPa]").next_to(y_axis, UP)
        y_label[1].scale(0.7)
        z_label = Tex("$\lambda_{y}$").move_to(z_axis).shift([-0.5,0,1.6])
        self.play(Create(x_axis), Create(y_axis))
        self.add_fixed_orientation_mobjects(x_label, y_label)
        self.play(FadeIn(x_label), FadeIn(y_label))
        self.slide_break()

        with open('manim_results_1.npy', 'rb') as f:
            [lmx, lmy, sgm, NODE, GOH, MR, HGO, Fung] = np.load(f)
        max_e = np.max(NODE)/2

        loadings = []
        indices = [0, 72, 72+76, 72+76+81, 72+76+81+101, 72+76+81+101+72] 
        for i in range(5):
            i1 = indices[i]
            i2 = indices[i+1]
            loadings.append([lmx[i1:i2], lmy[i1:i2], sgm[i1:i2], NODE[i1:i2], GOH[i1:i2], MR[i1:i2], HGO[i1:i2], Fung[i1:i2]])
        loadings = [loadings[0], loadings[2], loadings[1], loadings[3], loadings[4]]# Offy and equibiaxial need to switch places

        cmap = cm.get_cmap('RdYlGn_r') 
        

        #### Plot the Neural ODE graph
        # Plot just the strip-y dots first
        stry_dots = []
        lmx = loadings[4][0]
        lmy = loadings[4][1]
        sgm = loadings[4][2]
        err = loadings[4][3] 
        for i in range(0,lmx.shape[0],3):
            clr_rgb = cmap(err[i]/max_e)
            clr_hex = matplotlib.colors.rgb2hex(clr_rgb)
            dot = Dot(axis_node.coords_to_point(lmy[i], sgm[i], lmx[i])).scale(0.8).set_color(clr_hex)
            self.add_fixed_orientation_mobjects(dot)
            stry_dots.append(dot)
            self.play(FadeIn(dot), run_time=0.3)
        self.slide_break()
        self.play(FadeIn(cbar), FadeIn(cbar_label), FadeIn(cbar_tick1), FadeIn(cbar_tick2))
        self.slide_break()
        self.play(Create(z_axis))
        self.add_fixed_orientation_mobjects(z_label)
        self.play(FadeIn(z_label))

        # Plot the rest
        offx_dots = []
        equi_dots = []
        offy_dots = []
        strx_dots = []
        dots_list = [offx_dots, equi_dots, offy_dots, strx_dots]
        for lst, load in zip(dots_list, loadings[:4]):
            lmx = load[0]
            lmy = load[1]
            sgm = load[2]
            err = load[3]
            for i in range(0,err.shape[0],3):
                clr_rgb = cmap(err[i]/max_e)
                clr_hex = matplotlib.colors.rgb2hex(clr_rgb)
                dot = Dot(axis_node.coords_to_point(lmy[i], sgm[i], lmx[i])).scale(0.8).set_color(clr_hex)
                self.add_fixed_orientation_mobjects(dot)
                lst.append(dot)
        for dots in dots_list:
            self.remove(*dots)

        ang = 75
        unitv = np.array([np.cos(ang/180*np.pi), 0, np.sin(ang/180*np.pi)])
        self.move_camera(phi=-ang*DEGREES, theta=-0.5, gamma = 78*DEGREES) #visualize angles here: https://www.geogebra.org/m/hqPfxIpp
        for lst in dots_list:
            for dot in lst:
                self.play(FadeIn(dot), run_time=0.05)
        self.slide_break()

        # Wiggle the dots to show training and validation sets
        group1 = Group(*stry_dots, *equi_dots, *strx_dots)
        group2 = Group(*offx_dots, *offy_dots)
        self.play(ApplyWave(group1))
        self.slide_break()
        self.play(ApplyWave(group2))
        self.slide_break()

        #### Plot the GOH graph
        with open('manim_results_1.npy', 'rb') as f:
            [lmx, lmy, sgm, NODE, GOH, MR, HGO, Fung] = np.load(f)
        axis_GOH = axis_node.copy().move_to(ORIGIN).shift(30*RIGHT)
        x_axis_GOH = axis_GOH.get_x_axis()
        y_axis_GOH = axis_GOH.get_y_axis()
        z_axis_GOH = axis_GOH.get_z_axis()
        x_label_GOH = x_label.copy().next_to(x_axis_GOH, RIGHT)
        y_label_GOH = y_label.copy().next_to(y_axis_GOH, UP)
        z_label_GOH = z_label.copy().move_to(z_axis_GOH).shift([-0.5,0,1.6])
        label_GOH = Tex("GOH").move_to(z_axis_GOH).shift([-0.5,4,2.5])
        self.add(axis_GOH)
        self.add_fixed_orientation_mobjects(x_label_GOH, y_label_GOH, z_label_GOH, label_GOH)
        GOH_dots = []
        for i in range(0,lmx.shape[0],3):
            clr_rgb = cmap(GOH[i]/max_e)
            clr_hex = matplotlib.colors.rgb2hex(clr_rgb)
            dot = Dot(axis_GOH.coords_to_point(lmy[i], sgm[i], lmx[i])).scale(0.8).set_color(clr_hex)
            self.add_fixed_orientation_mobjects(dot)
            GOH_dots.append(dot)
        self.play(*[item.animate.shift(10*LEFT) for item in [axis_node, x_label, y_label, z_label, 
                                                            *offx_dots, *offy_dots, *equi_dots, 
                                                            *strx_dots, *stry_dots]],
                  *[item.animate.shift(30*LEFT) for item in [axis_GOH, x_label_GOH, y_label_GOH, z_label_GOH,
                                                            label_GOH, *GOH_dots]])
        self.slide_break()

        #### Plot the MR graph
        axis_MR = axis_node.copy().move_to(ORIGIN).shift(30*RIGHT)#.shift(6*unitv)
        x_axis_MR = axis_MR.get_x_axis()
        y_axis_MR = axis_MR.get_y_axis()
        z_axis_MR = axis_MR.get_z_axis()
        x_label_MR = x_label.copy().next_to(x_axis_MR, RIGHT)
        y_label_MR = y_label.copy().next_to(y_axis_MR, UP)
        z_label_MR = z_label.copy().move_to(z_axis_MR).shift([-0.5,0,1.6])
        label_MR = Tex("Mooney Rivlin").move_to(z_axis_MR).shift([-0.5,4,2.5])
        self.add(axis_MR)
        self.add_fixed_orientation_mobjects(x_label_MR, y_label_MR, z_label_MR, label_MR)
        MR_dots = []
        for i in range(0,lmx.shape[0],3):
            clr_rgb = cmap(MR[i]/max_e)
            clr_hex = matplotlib.colors.rgb2hex(clr_rgb)
            dot = Dot(axis_MR.coords_to_point(lmy[i], sgm[i], lmx[i])).scale(0.8).set_color(clr_hex)
            self.add_fixed_orientation_mobjects(dot)
            MR_dots.append(dot)
        self.play(*[item.animate.shift(10*LEFT) for item in [axis_GOH, x_label_GOH, y_label_GOH, z_label_GOH, label_GOH, *GOH_dots]],
                  *[item.animate.shift(30*LEFT) for item in [axis_MR, x_label_MR, y_label_MR, z_label_MR, label_MR, *MR_dots]])
        self.slide_break()

        #### Plot the HGO graph
        axis_HGO = axis_node.copy().move_to(ORIGIN).shift(30*RIGHT)#.shift(6*unitv)
        x_axis_HGO = axis_HGO.get_x_axis()
        y_axis_HGO = axis_HGO.get_y_axis()
        z_axis_HGO = axis_HGO.get_z_axis()
        x_label_HGO = x_label.copy().next_to(x_axis_HGO, RIGHT)
        y_label_HGO = y_label.copy().next_to(y_axis_HGO, UP)
        z_label_HGO = z_label.copy().move_to(z_axis_HGO).shift([-0.5,0,1.6])
        label_HGO = Tex("HGO").move_to(z_axis_HGO).shift([-0.5,4,2.5])
        self.add(axis_HGO)
        self.add_fixed_orientation_mobjects(x_label_HGO, y_label_HGO, z_label_HGO, label_HGO)
        HGO_dots = []
        for i in range(0,lmx.shape[0],3):
            clr_rgb = cmap(HGO[i]/max_e)
            clr_hex = matplotlib.colors.rgb2hex(clr_rgb)
            dot = Dot(axis_HGO.coords_to_point(lmy[i], sgm[i], lmx[i])).scale(0.8).set_color(clr_hex)
            self.add_fixed_orientation_mobjects(dot)
            HGO_dots.append(dot)
        self.play(*[item.animate.shift(10*LEFT) for item in [axis_MR, x_label_MR, y_label_MR, z_label_MR, label_MR, *MR_dots]],
                  *[item.animate.shift(30*LEFT) for item in [axis_HGO, x_label_HGO, y_label_HGO, z_label_HGO, label_HGO, *HGO_dots]])
        self.slide_break()

        #### Plot the Fung graph
        axis_Fung = axis_node.copy().move_to(ORIGIN).shift(30*RIGHT)#.shift(6*unitv)
        x_axis_Fung = axis_Fung.get_x_axis()
        y_axis_Fung = axis_Fung.get_y_axis()
        z_axis_Fung = axis_Fung.get_z_axis()
        x_label_Fung = x_label.copy().next_to(x_axis_Fung, RIGHT)
        y_label_Fung = y_label.copy().next_to(y_axis_Fung, UP)
        z_label_Fung = z_label.copy().move_to(z_axis_Fung).shift([-0.5,0,1.6])
        label_Fung = Tex("Fung").move_to(z_axis_Fung).shift([-0.5,4,2.5])
        self.add(axis_Fung)
        self.add_fixed_orientation_mobjects(x_label_Fung, y_label_Fung, z_label_Fung, label_Fung)
        Fung_dots = []
        for i in range(0,lmx.shape[0],3):
            clr_rgb = cmap(Fung[i]/max_e)
            clr_hex = matplotlib.colors.rgb2hex(clr_rgb)
            dot = Dot(axis_Fung.coords_to_point(lmy[i], sgm[i], lmx[i])).scale(0.8).set_color(clr_hex)
            self.add_fixed_orientation_mobjects(dot)
            Fung_dots.append(dot)
        self.play(*[item.animate.shift(10*LEFT) for item in [axis_HGO, x_label_HGO, y_label_HGO, z_label_HGO, label_HGO, *HGO_dots]],
                  *[item.animate.shift(30*LEFT) for item in [axis_Fung, x_label_Fung, y_label_Fung, z_label_Fung, label_Fung, *Fung_dots]])
        self.slide_break()
        self.play(*[item.animate.shift(10*LEFT) for item in [axis_Fung, x_label_Fung, y_label_Fung, z_label_Fung, label_Fung, *Fung_dots]],
                    FadeOut(cbar), FadeOut(cbar_label), FadeOut(cbar_tick1), FadeOut(cbar_tick2))

class Results_p3(SlideScene):
    def construct(self):
        heading = toc[3].copy().move_to(ORIGIN).scale(1.25).to_corner(UP)
        self.add(heading)

        axes = Axes(
            [0, 12], 
            [-0.025, 0.175, 0.05], 
            x_length=17, 
            y_axis_config={'unit_size':0.05,
                           'include_numbers':True,
                           'font_size':60},
            x_axis_config={'include_ticks':False}
            ).scale(0.6).move_to(ORIGIN)

        y_axis = axes.get_y_axis()
        x_axis = axes.get_x_axis()
        ylabel = Tex("Error", " [MPa]").rotate(90*DEGREES).next_to(y_axis,LEFT)
        x_axis.align_to(y_axis,DOWN)
        self.play(Create(axes), Write(ylabel))

        labels = [Tex('GOH'), Tex('MR'), Tex('HGO'), Tex('Fung'), Tex('N-ODE')]
        with open('manim_results_1.npy', 'rb') as f:
            [lmx, lmy, sgm, NODE, GOH, MR, HGO, Fung] = np.load(f)
        
        #### Training box plots
        medians = [np.median(GOH[148:]), np.median(MR[148:]), np.median(HGO[148:]), np.median(Fung[148:]), np.median(NODE[148:])]
        lower_quartiles = [np.quantile(GOH[148:],0.25), np.quantile(MR[148:],0.25), np.quantile(HGO[148:],0.25), np.quantile(Fung[148:],0.25), np.quantile(NODE[148:],0.25)]
        upper_quartiles = [np.quantile(GOH[148:],0.75), np.quantile(MR[148:],0.75), np.quantile(HGO[148:],0.75), np.quantile(Fung[148:],0.75), np.quantile(NODE[148:],0.75)]
        trn_median_bars = []
        trn_boxes_upper = []
        trn_boxes_lower = []
        trn_wh_body_upper = []
        trn_wh_body_lower = []
        trn_wh_ends_upper = []
        trn_wh_ends_lower = []
        x_locs = np.linspace(1,11, 5)
        iterables = zip(x_locs, labels, medians, lower_quartiles, upper_quartiles)
        for x, label, median, lowerq, upperq in iterables:
            label.move_to(axes.coords_to_point(x, -0.05))

            median_bar_end1 = axes.coords_to_point(x-0.5-0.25, median, 0)
            median_bar_end2 = axes.coords_to_point(x-0.5+0.25, median, 0)
            median_bar_cent = axes.coords_to_point(x-0.5, median, 0)
            median_bar = Line(median_bar_end1, median_bar_end2).set_color(BLUE)
            trn_median_bars.append(median_bar)

            upperqy = axes.coords_to_point(x-0.5,upperq)[1]
            lowerqy = axes.coords_to_point(x-0.5,lowerq)[1]

            box1_height = upperqy-median_bar_cent
            box1_height = box1_height[1]
            box1 = Rectangle(color=BLUE, height=box1_height, width=0.32).move_to(median_bar_cent).align_to(median_bar, DOWN)
            trn_boxes_upper.append(box1)

            box2_height = median_bar_cent-lowerqy
            box2_height = box2_height[1]
            box2 = Rectangle(color=BLUE, height=box2_height, width=0.32).move_to(median_bar_cent).align_to(median_bar, UP)
            trn_boxes_lower.append(box2)

            #Whiskers
            wh1_lowerend = axes.coords_to_point(x-0.5,upperq)
            wh1_upperend = axes.coords_to_point(x-0.5,upperq+1.5*(upperq-lowerq))
            wh2_lowerend = axes.coords_to_point(x-0.5,np.max([0,lowerq-1.5*(upperq-lowerq)]))
            wh2_upperend = axes.coords_to_point(x-0.5,lowerq)

            wh1_body = Line(wh1_lowerend, wh1_upperend).set_color(BLUE)
            wh2_body = Line(wh2_upperend, wh2_lowerend).set_color(BLUE)

            trn_wh_body_upper.append(wh1_body)
            trn_wh_body_lower.append(wh2_body)

            p1 = axes.coords_to_point(x-0.5-0.1, upperq+1.5*(upperq-lowerq))
            p2 = axes.coords_to_point(x-0.5+0.1, upperq+1.5*(upperq-lowerq))
            wh1_end = Line(p1, p2).set_color(BLUE)

            p1 = axes.coords_to_point(x-0.5-0.1, np.max([0,lowerq-1.5*(upperq-lowerq)]))
            p2 = axes.coords_to_point(x-0.5+0.1, np.max([0,lowerq-1.5*(upperq-lowerq)]))
            wh2_end = Line(p1, p2).set_color(BLUE)

            trn_wh_ends_upper.append(wh1_end)
            trn_wh_ends_lower.append(wh2_end)


        trn_legend_bar = Line(axes.coords_to_point(8,0.17), axes.coords_to_point(8.5, 0.17)).set_color(BLUE)
        trn_legend_tex = Tex("Training").next_to(trn_legend_bar,RIGHT)

        self.play(*[FadeIn(label) for label in labels])
        self.play(FadeIn(trn_legend_bar), FadeIn(trn_legend_tex))
        self.play(*[Create(item) for item in trn_median_bars])
        self.play(*[GrowFromEdge(item, DOWN) for item in trn_boxes_upper], *[GrowFromEdge(item, UP) for item in trn_boxes_lower])
        self.play(*[Create(item) for item in trn_wh_body_upper], *[Create(item) for item in trn_wh_body_lower])
        self.play(*[Create(item) for item in trn_wh_ends_upper], *[Create(item) for item in trn_wh_ends_lower])
        self.slide_break()

        #### Validation box plots
        medians = [np.median(GOH[:148]), np.median(MR[:148]), np.median(HGO[:148]), np.median(Fung[:148]), np.median(NODE[:148])]
        lower_quartiles = [np.quantile(GOH[:148],0.25), np.quantile(MR[:148],0.25), np.quantile(HGO[:148],0.25), np.quantile(Fung[:148],0.25), np.quantile(NODE[:148],0.25)]
        upper_quartiles = [np.quantile(GOH[:148],0.75), np.quantile(MR[:148],0.75), np.quantile(HGO[:148],0.75), np.quantile(Fung[:148],0.75), np.quantile(NODE[:148],0.75)]
        val_median_bars = []
        val_boxes_upper = []
        val_boxes_lower = []
        val_wh_body_upper = []
        val_wh_body_lower = []
        val_wh_ends_upper = []
        val_wh_ends_lower = []
        x_locs = np.linspace(1,11, 5)
        iterables = zip(x_locs, labels, medians, lower_quartiles, upper_quartiles)
        for x, label, median, lowerq, upperq in iterables:
            median_bar_end1 = axes.coords_to_point(x+0.5-0.25, median, 0)
            median_bar_end2 = axes.coords_to_point(x+0.5+0.25, median, 0)
            median_bar_cent = axes.coords_to_point(x+0.5, median, 0)
            median_bar = Line(median_bar_end1, median_bar_end2).set_color(GREEN)
            val_median_bars.append(median_bar)

            upperqy = axes.coords_to_point(x+0.5,upperq)[1]
            lowerqy = axes.coords_to_point(x+0.5,lowerq)[1]

            box1_height = upperqy-median_bar_cent
            box1_height = box1_height[1]
            box1 = Rectangle(color=GREEN, height=box1_height, width=0.32).move_to(median_bar_cent).align_to(median_bar, DOWN)
            val_boxes_upper.append(box1)

            box2_height = median_bar_cent-lowerqy
            box2_height = box2_height[1]
            box2 = Rectangle(color=GREEN, height=box2_height, width=0.32).move_to(median_bar_cent).align_to(median_bar, UP)
            val_boxes_lower.append(box2)

            #Whiskers
            wh1_lowerend = axes.coords_to_point(x+0.5,upperq)
            wh1_upperend = axes.coords_to_point(x+0.5,upperq+1.5*(upperq-lowerq))
            wh2_lowerend = axes.coords_to_point(x+0.5,np.max([0,lowerq-1.5*(upperq-lowerq)]))
            wh2_upperend = axes.coords_to_point(x+0.5,lowerq)

            wh1_body = Line(wh1_lowerend, wh1_upperend).set_color(GREEN)
            wh2_body = Line(wh2_upperend, wh2_lowerend).set_color(GREEN)

            val_wh_body_upper.append(wh1_body)
            val_wh_body_lower.append(wh2_body)

            p1 = axes.coords_to_point(x+0.5-0.1, upperq+1.5*(upperq-lowerq))
            p2 = axes.coords_to_point(x+0.5+0.1, upperq+1.5*(upperq-lowerq))
            wh1_end = Line(p1, p2).set_color(GREEN)

            p1 = axes.coords_to_point(x+0.5-0.1, np.max([0,lowerq-1.5*(upperq-lowerq)]))
            p2 = axes.coords_to_point(x+0.5+0.1, np.max([0,lowerq-1.5*(upperq-lowerq)]))
            wh2_end = Line(p1, p2).set_color(GREEN)

            val_wh_ends_upper.append(wh1_end)
            val_wh_ends_lower.append(wh2_end)

        val_legend_bar = Line(axes.coords_to_point(8,0.15), axes.coords_to_point(8.5, 0.15)).set_color(GREEN).next_to(trn_legend_bar,1.8*DOWN)
        val_legend_tex = Tex("Validation").next_to(val_legend_bar,RIGHT)

        self.play(FadeIn(val_legend_bar), FadeIn(val_legend_tex))
        self.play(*[Create(item) for item in val_median_bars])
        self.play(*[GrowFromEdge(item, DOWN) for item in val_boxes_upper], *[GrowFromEdge(item, UP) for item in val_boxes_lower])
        self.play(*[Create(item) for item in val_wh_body_upper], *[Create(item) for item in val_wh_body_lower])
        self.play(*[Create(item) for item in val_wh_ends_upper], *[Create(item) for item in val_wh_ends_lower])
        self.slide_break()


        # Remove everything
        self.play(*[FadeOut(obj) for obj in [axes, *labels, ylabel, *trn_legend_bar, *trn_legend_tex, *trn_median_bars, 
                                            *trn_boxes_upper, *trn_boxes_lower, *trn_wh_body_upper, *trn_wh_body_lower,
                                            *trn_wh_ends_upper, *trn_wh_ends_lower, *val_legend_bar, *val_legend_tex, *val_median_bars, 
                                            *val_boxes_upper, *val_boxes_lower, *val_wh_body_upper, *val_wh_body_lower,
                                            *val_wh_ends_upper, *val_wh_ends_lower]])
        self.slide_break()




        #### 80-20 split
        x_range = [1.0, 1.17, 0.05]
        y_range = [  0,  1.5,  0.5]
        axes = Axes(
            x_range, 
            y_range, 
            x_length=12, 
            axis_config={'include_numbers':True,
                         'font_size':70}
            ).scale(0.7).move_to(ORIGIN).shift(DOWN)
        x_axis = axes.get_x_axis()
        y_axis = axes.get_y_axis()
        x_label = Tex("$\lambda_{x}$").next_to(x_axis, RIGHT).scale(1.2)
        y_label = Tex("$\mathbf{\sigma}_{1}$", "[MPa]").next_to(y_axis, UP).scale(1.2)
        y_label[1].scale(0.7)
        self.play(Create(x_axis), Create(y_axis))
        self.play(FadeIn(x_label), FadeIn(y_label))

        with open('manim_results_1.npy', 'rb') as f:
            [lmx, lmy, sgm, NODE, GOH, MR, HGO, Fung] = np.load(f)

        loadings = []
        indices = [0, 72, 72+76, 72+76+81, 72+76+81+101, 72+76+81+101+72] 
        for i in range(5):
            i1 = indices[i]
            i2 = indices[i+1]
            loadings.append([lmx[i1:i2], lmy[i1:i2], sgm[i1:i2], NODE[i1:i2], GOH[i1:i2], MR[i1:i2], HGO[i1:i2], Fung[i1:i2]])
        loadings = [loadings[0], loadings[2], loadings[1], loadings[3], loadings[4]]# Offy and equibiaxial need to switch places

        trn_dots = []
        val_dots = []
        lmx = loadings[4][0]
        lmy = loadings[4][1]
        sgm = loadings[4][2]
        i80 = 0
        for i in range(0,lmx.shape[0],2):
            if i < lmx.shape[0]*0.8:
                dot = Dot(axes.coords_to_point(lmy[i], sgm[i], lmx[i])).scale(0.8).set_color(BLUE)
                trn_dots.append(dot)
                i80 = i
            else:
                dot = Dot(axes.coords_to_point(lmy[i], sgm[i], lmx[i])).scale(0.8).set_color(GREEN)
                val_dots.append(dot)
        trn_tex = Tex("80\% Training").set_color(BLUE).move_to(axes.coords_to_point(1.06,1.2))
        val_tex = Tex("20\% Validation").set_color(GREEN).move_to(axes.coords_to_point(1.165,1.2))
        x_sep = (lmy[i80]+lmy[i80+2])/2
        sep = axes.get_vertical_line(axes.coords_to_point(x_sep, 1.3), line_config={"dashed_ratio": 0.5})
        self.play(*[FadeIn(dot) for dot in trn_dots])
        self.play(FadeIn(trn_tex), FadeIn(sep))
        self.slide_break()
        self.play(*[FadeIn(dot) for dot in val_dots])
        self.play(FadeIn(val_tex))
        self.slide_break()

        self.play(*[FadeOut(dot) for dot in [axes, *trn_dots, *val_dots, trn_tex, val_tex, sep, x_label, y_label]])

        #### 80-20 Table
        tbl = DecimalTable([[0.083, 0.114, 0.245, 0.246, 0.009],
                            [0.010, 0.024, 0.071, 0.123, 0.062],
                            [0.037, 0.100, 0.141, 0.035, 0.031],
                            [0.056, 0.123, 0.051, 0.133, 0.098],
                            [0.120, 0.095, 0.051, 0.194, 0.038],
                            [0.062, 0.091, 0.112, 0.146, 0.048]],
                            col_labels = labels,
                            row_labels = [Tex("Off-x"), Tex("Off-y"), Tex("Equi"), Tex("Strip-x"), Tex("Strip-y"), Tex("Average")],
                            element_to_mobject_config={"num_decimal_places": 3}
                            ).scale(0.6).shift(DOWN)
        tbl_label = Tex("Validation Error", "[MPa]").next_to(tbl,UP)
        tbl_label[1].scale(0.8)
        mincells = []
        mincells.append(tbl.get_cell((2,6), color=GREEN))
        mincells.append(tbl.get_cell((3,2), color=GREEN))
        mincells.append(tbl.get_cell((4,6), color=GREEN))
        mincells.append(tbl.get_cell((5,4), color=GREEN))
        mincells.append(tbl.get_cell((6,6), color=GREEN))
        av_min = tbl.get_cell((7,6), color=GREEN).set_fill(opacity=0.5)
        mincells = [obj.set_fill(opacity=0.5) for obj in mincells]
        first_ver_line = tbl.get_vertical_lines()[0].set_color(YELLOW)
        first_hor_line = tbl.get_horizontal_lines()[0].set_color(YELLOW)
        last_hor_line  = tbl.get_horizontal_lines()[-1].set_color(YELLOW)
        self.play(FadeIn(tbl), FadeIn(tbl_label))
        self.slide_break()
        self.play(*[Create(obj) for obj in mincells])
        self.slide_break()
        self.play(Create(av_min))
        self.slide_break()
        self.play(*[FadeOut(obj) for obj in [*mincells, av_min, tbl, tbl_label, heading]])

class FEM_p1(SlideScene):
    def construct(self):
        self.play(FadeIn(toc))
        self.slide_break()

        heading = toc[4].copy().move_to(ORIGIN).scale(1.25).to_corner(UP)
        self.play(FadeOut(toc), ReplacementTransform(toc[4],heading))
        self.slide_break()
        twoDfem = Tex("Custom 2D FE solver").move_to(2.2*UP)
        self.play(Write(twoDfem))
        self.slide_break()
        uni = SVGMobject("2DFEM/Uniaxial.svg").scale(2).move_to(3*LEFT+DOWN)
        shr = SVGMobject("2DFEM/Shear.svg").scale(2).move_to(3*RIGHT+DOWN)
        self.play(Write(uni), Write(shr))
        self.slide_break()
        self.play(FadeOut(uni, shr, twoDfem))

        simulia=SVGMobject("3ds_logo_short.svg").scale(0.3).shift(LEFT*1.7+UP*2.2)
        abaqus = Tex("ABAQUS", font_size=60).next_to(simulia, RIGHT)
        self.play(Write(simulia), Write(abaqus))
        self.slide_break()

        umat = Tex("UMAT").move_to(LEFT*3.5+DOWN).scale(0.6)
        umatbox = Rectangle(height=1.2, width=3.0).move_to(umat)
        uani = Tex("UANISOHYPER").move_to(RIGHT*3.5+DOWN).scale(0.6)
        uanibox = Rectangle(height=1.2, width=3.0).move_to(uani)
        self.play(Write(umat), Write(umatbox))
        self.play(Write(uani), Write(uanibox))
        self.slide_break()

        umatinarr = Arrow(start=umatbox.get_top()+[0,1,0], end=umatbox.get_top())
        umatin = Group(
            Tex("$\mathbf{F}_t$"),
            Tex("$\mathbf{F}_{t+\Delta t}$"),
            Tex("Material properties")
        ).arrange(DOWN, buff=0.2).scale(0.8).next_to(umatinarr, UP)
        self.play(Create(umatinarr), Write(umatin[0]), Write(umatin[1]), Write(umatin[2]))
        self.slide_break()
        
        umatoutarr = Arrow(start=umatbox.get_bottom(), end=umatbox.get_bottom()+[0,-1,0])
        umatout = Group(
            Tex("Stress (Cauchy)"),
            Tex("Tangent stiffness matrix")
        ).arrange(DOWN, buff=0.2).scale(0.8).next_to(umatoutarr, DOWN)
        self.play(Create(umatoutarr), Write(umatout[0]), Write(umatout[1]))
        self.slide_break()

        uaniinarr = Arrow(start=uanibox.get_top()+[0,1,0], end=uanibox.get_top())
        uaniin = Group(
            Tex("Invariants"),
            Tex("Number of fiber families"),
            Tex("Material properties")
        ).arrange(DOWN, buff=0.2).scale(0.8).next_to(uaniinarr, UP)
        self.play(Create(uaniinarr), Write(uaniin[0]), Write(uaniin[1]), Write(uaniin[2]))
        self.slide_break()

        uanioutarr = Arrow(start=uanibox.get_bottom(), end=uanibox.get_bottom()+[0,-1,0])
        uaniout = MathTex("\Psi, \, \, \partial \Psi / \partial \\bar{I}_i, \, \
        \, \partial^2 \Psi / \partial \\bar{I}_i \\bar{I}_j").scale(0.8).next_to(uanioutarr, DOWN)
        self.play(Create(uanioutarr), Write(uaniout))
        self.slide_break()

        self.play(FadeOut(simulia, abaqus, umat, umatbox, uani, uanibox, 
                          umatin, umatinarr, umatoutarr, umatout,
                          uaniin, uaniinarr, uanioutarr, uaniout))
        self.slide_break()


        # cbar = SVGMobject("cbar_FEM.svg").to_edge(LEFT)

        # limits = [[0.00, 0.34], [0.00, 1.40], [0.00, 0.42], [0.00, 0.13], [0.00, 8.00]] #cranium max: 80.11
        # legend = Tex("$\sigma_1$ [MPa]").next_to(cbar)
        # lims = Group(
        #     Tex(str(limits[0][1])),
        #     Tex(str(limits[0][0]))
        # ).arrange(DOWN, aligned_edge=LEFT, buff=1.8).next_to(cbar)

        # self.play(Write(cbar))
        # self.play(Write(legend), Write(lims[0]), Write(lims[1]))
        # self.slide_break()

        # self.play(Unwrite(lims[0]), Unwrite(lims[1]))
        # lims = Group(
        #     Tex(str(limits[1][1])),
        #     Tex(str(limits[1][0]))
        # ).arrange(DOWN, aligned_edge=LEFT, buff=1.8).next_to(cbar)
        # self.slide_break()
        # self.play(Write(lims[0]), Write(lims[1]))
        # self.slide_break()

        # self.play(Unwrite(lims[0]), Unwrite(lims[1]))
        # lims = Group(
        #     Tex(str(limits[2][1])),
        #     Tex(str(limits[2][0]))
        # ).arrange(DOWN, aligned_edge=LEFT, buff=1.8).next_to(cbar)
        # self.slide_break()
        # self.play(Write(lims[0]), Write(lims[1]))
        # self.slide_break()

        # self.play(Unwrite(lims[0]), Unwrite(lims[1]))
        # lims = Group(
        #     Tex(str(limits[3][1])),
        #     Tex(str(limits[3][0]))
        # ).arrange(DOWN, aligned_edge=LEFT, buff=1.8).next_to(cbar)
        # self.slide_break()
        # self.play(Write(lims[0]), Write(lims[1]))
        # self.slide_break()

        # # Fade Out the colorbar for the introduction of the scalp FEM
        # self.play(FadeOut(lims), FadeOut(cbar), FadeOut(legend))
        # self.slide_break()

        # pre  = ImageMobject('surg_pre.jpg').scale(0.2).shift(LEFT*2)
        # arr = MathTex(r"\rightarrow").next_to(pre)
        # post = ImageMobject('surg_post.jpg').scale(0.2).next_to(arr)
        # self.play(FadeIn(pre))
        # self.play(FadeIn(arr))
        # self.play(FadeIn(post))
        # self.slide_break()
        # self.play(FadeOut(pre), FadeOut(post), FadeOut(arr))


        # self.slide_break()
        # lims = Group(
        #     Tex(str(limits[4][1]), r" $\uparrow$"),
        #     Tex(str(limits[4][0]))
        # ).arrange(DOWN, aligned_edge=LEFT, buff=1.8).next_to(cbar)
        # self.play(FadeIn(lims), FadeIn(cbar), FadeIn(legend))

        # self.slide_break()
        # self.play(FadeOut(lims), FadeOut(cbar), FadeOut(legend), FadeOut(heading))

class FEM_p3(SlideScene):
    def construct(self):
        heading = toc[4].copy().move_to(ORIGIN).scale(1.25).to_corner(UP)
        self.add(heading)

        convax = SVGMobject('convergence_axes.svg').scale(3).shift(0.5*DOWN)
        conv = SVGMobject('convergence.svg').scale(3).shift(0.5*DOWN)
        self.play(Write(convax))
        self.play(Write(conv))
        self.slide_break()

class Conc(SlideScene):
    def construct(self):
        self.play(FadeIn(toc))
        self.slide_break()

        heading = toc[5].copy().move_to(ORIGIN).scale(1.25).to_corner(UP)
        self.play(FadeOut(toc), ReplacementTransform(toc[5],heading))
        self.slide_break()

        conc1 = Tex("$\\bullet$ Polyconvexity using Neural ODEs").shift(2*UP+1.5*LEFT)
        conc1exp = Group(
            Tex("$\\bullet$ The right kind of convexity" ),
            Tex("$\\bullet$ Guaranteed everywhere" ),
            Tex("$\\bullet$ Better convergence and fewer computations")
        ).arrange(DOWN,aligned_edge=LEFT,buff=0.35).next_to(conc1,DOWN, aligned_edge=LEFT).shift(RIGHT*0.1).scale(0.9)
        conc23 = Group(
            Tex("$\\bullet$ Superior predictive capabilities when trained \\\ with experimental data"),
            Tex("$\\bullet$ Implementation in Abaqus")
        ).arrange(DOWN,aligned_edge=LEFT,buff=0.35).next_to(conc1,DOWN, aligned_edge=LEFT).shift(DOWN*2.3)
        self.play(FadeIn(conc1), shift=0.5*UP)
        for i in range(len(conc1exp)):
            self.slide_break()
            self.play(FadeIn(conc1exp[i], shift=0.5*UP))
        for i in range(len(conc23)):
            self.slide_break()
            self.play(FadeIn(conc23[i], shift=0.5*UP))
        self.slide_break()

        self.play(FadeOut(conc1, conc1exp, conc23))
        ideas = Group(
            Tex("Major ideas:"),
            Tex("$\\circ$ Knowing that polyconvexity is needed"),
            Tex("$\\circ$ Knowing what classes of functions are polyconvex"),
            Tex("$\\circ$ Realizing that N-ODEs can be used to build convex functions in one variable"),
            Tex("$\\circ$ Making sure that the model can capture basic closed form models exactly"),
            Tex("$\\odot$ Knowing that derivatives are better to work with, than the energy"),
            Tex("$\\odot$ Knowing how to handle compressible and nearly incompressible cases"),
            Tex("$\\odot$ Accounting for interactions between invariants via the $\Psi_{I_i+I_j}(I_i + I_j)$ terms"),
            Tex("$\\bullet$ To make sure that the model is indeed hyperelastic, i.e., the derivative functions do come from the same energy function"),
            Tex("$\\bullet$ Ensuring stress-free state when there is no deformation"),
            Tex("$\\bullet$ Training in the log space to capture response in small deformations"),
        ).arrange(DOWN, aligned_edge=LEFT,buff=0.35).scale(0.7).shift(0.5*DOWN)

        
        

        for i in range(len(ideas)):
            self.slide_break()
            self.play(FadeIn(ideas[i], shift=0.5*LEFT))
        self.slide_break()

        self.play(FadeOut(heading, ideas))
        twitter_logo = SVGMobject("Twitter-logo.svg").scale(0.25).shift(LEFT)
        twitter_addr = Text("@tajtac").next_to(twitter_logo).shift(0.05*DOWN)
        self.play(Write(twitter_logo))
        self.play(Write(twitter_addr))