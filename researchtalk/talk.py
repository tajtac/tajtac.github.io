from manim_slide import DOWN, LEFT, Group, Tex, SlideScene, SVGMobject, FadeIn, Write, Unwrite, FadeOut, YELLOW, WHITE, UP, BLACK, Text, RIGHT, DEGREES, GREEN
from manim_slide import ReplacementTransform, ORIGIN, TexTemplate, MathTex, Circumscribe, Transform, BLUE, Dot, Create, Flash, Restore, Arrow, Square
from manim_slide import Axes, DashedLine, SlideMovingCameraScene, Cross, ShowPassingFlash, SlideThreeDScene, ImageMobject, ThreeDAxes, ApplyWave
from manim_slide import Line, Rectangle, GrowFromEdge, BLUE_C, rate_functions, linear, MathTex
import numpy as np
from pylab import cm
import matplotlib, pickle
from jax.flatten_util import ravel_pytree

#Define some shades of orange
orange1 = "#fc8d62"
orange2 = "#E8B094"
orange3 = "#F25106"
orange4 = "#8F4C2D"
orange5 = "#FEC2AD"
orange6 = "#CAAA9F"

long_tex_templ = TexTemplate()
long_tex_templ.add_to_preamble(r"\usepackage[none]{hyphenat}")


toctitle = Tex('Research areas').shift(1.5*UP)
toc = Group(
    Tex("1. Data-driven hyperelasticity"),
    Tex("2. Inelasticity - ", "Viscoelasticity", ", ", "Damage"),
    Tex("3. Uncertainty quantification - Generative hyperelasticity")
).arrange(DOWN,aligned_edge=LEFT,buff=0.4).scale(0.75)

class Title(SlideScene):
    def construct(self):
        title = Tex(r'Physics-constrained, data-driven modeling \\ of elastic and inelastic material behavior').shift(2.5*UP)
        name = Tex(r'Vahidullah Tac').scale(0.75)
        # collab = Tex('A. Buganza Tepole, F. Sahli Costabal').scale(0.8).next_to(name,DOWN).shift(0.3*DOWN)
        purdue=SVGMobject("purdue_logo.svg").shift(2.5*DOWN).scale(1/4)#.next_to(1.5*DOWN,LEFT,buff=2.5)

        self.play(FadeIn(name))
        self.slide_break()
        self.play(FadeIn(title))
        self.slide_break()
        # self.play(FadeIn(collab))
        self.play(Write(purdue))
        self.slide_break()

        self.play(FadeOut(name, title, purdue))
        self.slide_break()

        self.play(FadeIn(toc, toctitle))
        self.slide_break()

        self.play(toc[0].animate.scale(1.2).set_color(YELLOW))

        for i in range(1, len(toc)):
            self.slide_break()
            self.play(toc[i].animate.scale(1.2).set_color(YELLOW),toc[i-1].animate.scale(1/1.2).set_color(WHITE))
            
        self.slide_break()
        self.play(toc[-1].animate.scale(1/1.2).set_color(WHITE))

class hyper_p1_nnmat(SlideScene):
    def construct(self):
        self.add(toc, toctitle)

        heading = toc[0].copy().move_to(ORIGIN).scale(1.25).to_corner(UP)
        self.play(FadeOut(toc, toctitle), ReplacementTransform(toc[0],heading))
        self.slide_break()

        whatishyper = Tex('What is hyperelasticity?').move_to(2*UP + 4*LEFT)
        self.play(Write(whatishyper))
        self.slide_break()

        # defhyper = MathTex(r'\mathbf{P}', ' = {', '{\partial \Psi}', ' \over ', '{\partial \mathbf{F}}', '}')
        # defhyper = MathTex('\mathbf{P}', ' = {{\partial ', '\Psi', '} \over {\partial ', '\mathbf{F}', '}}')
        defhyper = MathTex('\mathbf{P}', ' = {', '{\partial ', '\Psi', '(\mathbf{F})} \over ', '{\partial ', '\mathbf{F}}', '}')
        definitions1 = Group(
            Tex('$\\mathbf{P}$', ': First Piola-Kirchhoff stress'),
            Tex('$\mathbf{F}$', ': Deformation gradient'),
            Tex('$\Psi$', ': Strain Energy Density Function (SEDF)')
        ).arrange(DOWN,aligned_edge=LEFT,buff=0.4).scale(0.75).next_to(defhyper, DOWN).shift(3*LEFT+DOWN)
        defhyper_copy = defhyper.copy()
        self.play(Write(defhyper)) 
        self.slide_break()

        self.play(Transform(defhyper_copy[0], definitions1[0][0]))
        self.play(Write(definitions1[0][1]))
        self.slide_break()
        self.play(Transform(defhyper_copy[6], definitions1[1][0]))
        self.play(Write(definitions1[1][1]))
        self.slide_break()
        self.play(Transform(defhyper_copy[3], definitions1[2][0]))
        self.play(Write(definitions1[2][1]))
        self.slide_break()

        self.play(Unwrite(defhyper), FadeOut(definitions1), FadeOut(defhyper_copy))

        eqn_S = MathTex('\mathbf{S}', ' = 2{', '{\partial ', '\Psi', '(\mathbf{C})} \over ', '{\partial ', '\mathbf{C}}', '}')
        definitions2 = Group(
            Tex('$\\mathbf{S}$', ': Second Piola-Kirchhoff stress'),
            Tex('$\mathbf{C}$', ': Right Cauchy-Green deformation tensor'),
            Tex('$\Psi$', ': Strain Energy Density Function (SEDF)')
        ).arrange(DOWN,aligned_edge=LEFT,buff=0.4).scale(0.75).next_to(defhyper, DOWN).shift(3*LEFT+DOWN)
        eqn_S_copy = eqn_S.copy()
        self.play(Write(eqn_S)) 
        self.slide_break()

        self.play(Transform(eqn_S_copy[0], definitions2[0][0]))
        self.play(Write(definitions2[0][1]))
        self.slide_break()
        self.play(Transform(eqn_S_copy[6], definitions2[1][0]))
        self.play(Write(definitions2[1][1]))
        self.play(Transform(eqn_S_copy[3], definitions2[2][0]))
        self.play(Write(definitions2[2][1]))
        self.slide_break()

        self.play(Unwrite(eqn_S), FadeOut(definitions2), FadeOut(eqn_S_copy), FadeOut(whatishyper))

        # dPsi = Tex('$\\frac{\partial \Psi}{\partial I_1}, \\frac{\partial \P rtial I_{4v}}, \\frac{\partial \Psi}{\partial I_{4w}}, \cdots$')
        # self.play(Write(dPsi))
        # self.slide_break()
        # self.play(FadeOut(dPsi, whatishyper))


        # List of names
        names = ["Neo Hookean Model",
                 "Yeoh Model",
                 "Gent Model",
                 "Yeoh-Fleming Model",
                 "Two-Term Model",
                 "Exp-ln Model",
                 "Mooney Model",
                 "Isihara Model",
                 "Biderman Model",
                 "Gent-Thomas Model",
                 "Hart-Smith Model",
                 "Alexander Model",
                 "James Model",
                 "Haines-Wilson Model",
                 "Swanson Model",
                 "Kilian (Van Der Waals) Model",
                 "Yamashita-Kawabata Model",
                 "Lion Model",]
                #  "Diani-Rey Model",
                #  "Haupt-Sedlan Model",
                #  "Chevalier-Marco Model",
                #  "Pucci-Saccomandi Model",
                #  "Amin Model",
                #  "Beda Model",
                #  "Carroll Model",
                #  "Nunes Model",
                #  "Yaya-Bechir Model",
                #  ]
        names = [Tex(name) for name in names]

        self.play(FadeIn(names[0], shift=UP))
        for i in range(len(names)-1):
            self.play(FadeOut(names[i], shift=UP), FadeIn(names[i+1], shift=UP), run_time=0.7)
        self.slide_break()


        names = names[::-1]
        for i in range(len(names)-1):
            self.play(FadeOut(names[i], shift=DOWN), FadeIn(names[i+1], shift=DOWN), run_time=0.1)
        ml = Tex('Machine Learning?')
        ml.set_color(YELLOW)
        self.play(FadeOut(names[-1], shift=DOWN), FadeIn(ml, shift=DOWN), run_time=1.0)
        self.slide_break()

        
        self.play(FadeOut(ml))
        nnmat_title = Tex("\\begin{tabular}{c}Data-driven modeling of the mechanical behavior of anisotropic soft biological tissue\\textsuperscript{1}\\end{tabular}").scale(0.7).move_to(2.5*UP)
        nnmat_cite = Tex("\\begin{tabular}{c}\\textsuperscript{1}V. Tac, V.D. Sree, F. Sahli Costabal, A. Buganza Tepole, Engineering with Computers, 2022.\\end{tabular}").scale(0.5).move_to(3.5*DOWN)
        self.play(Write(nnmat_title))
        self.play(FadeIn(nnmat_cite))
        self.slide_break()

        nn_inps = Group(
            Tex('$I_1$'),
            Tex('$I_2$'),
            Tex('$I_{4v}$'),
            Tex('$I_{4w}$')
        ).arrange(RIGHT, buff=0.8).scale(0.75).set_color(YELLOW)
        self.play(FadeIn(nn_inps))
        self.slide_break()

        self.play(nn_inps.animate.arrange(DOWN, buff=0.3))
        self.play(nn_inps.animate.move_to(4.1*LEFT))

        nn_diag = SVGMobject('nn.svg').scale(2).rotate(180*DEGREES)
        self.play(Write(nn_diag))

        # nn_outs = Group(
        #     Tex('$\Psi_1$'),
        #     Tex('$\Psi_2$'),
        #     Tex('$\Psi_{4v}$'),
        #     Tex('$\Psi_{4w}$')
        # ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).scale(0.75).move_to(4.2*RIGHT).set_color(YELLOW)
        nn_outs = Group(
            Tex('$\partial \Psi / \partial I_1$'),
            Tex('$\partial \Psi / \partial I_2$'),
            Tex('$\partial \Psi / \partial I_{4v}$'),
            Tex('$\partial \Psi / \partial I_{4w}$')
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).scale(0.75).move_to(4.5*RIGHT).set_color(YELLOW)
        self.play(FadeIn(nn_outs))
        self.slide_break()

        self.play(FadeOut(nn_diag, nn_inps))
        self.play(nn_outs.animate.move_to(4*LEFT))
        arr = Tex('$\\rightarrow$').next_to(nn_outs, RIGHT)
        sgm = Tex('$\mathbf{S}$').next_to(arr, RIGHT)
        chkmrk1 = Tex('$\checkmark$').next_to(sgm, RIGHT).set_color(GREEN)
        self.play(FadeIn(arr, sgm))
        self.play(Write(chkmrk1))
        self.slide_break()

        CC = Tex(', $\mathbb{C}$ $\checkmark$').next_to(chkmrk1, RIGHT)
        CC[0][-1].set_color(GREEN)
        self.play(FadeIn(CC))
        self.slide_break()

        scale = 3
        shift = 0.5*DOWN
        self.play(FadeOut(nn_outs, arr, sgm, chkmrk1, CC))
        porc_stp1 = SVGMobject('fig_porcine_stp1.svg').scale(scale).shift(shift)
        self.play(Write(porc_stp1))
        self.slide_break()

        porc_stp2 = SVGMobject('fig_porcine_stp2.svg').scale(scale).shift(shift)
        self.play(Write(porc_stp2))
        porc_stp3 = SVGMobject('fig_porcine_stp3.svg').scale(scale).shift(shift)
        self.play(Write(porc_stp3))
        self.slide_break()


        porc_stp4 = SVGMobject('fig_porcine_stp4.svg').scale(scale).shift(shift)
        self.play(Write(porc_stp4))
        self.slide_break()


        porc_stp2.set_opacity(0.1)
        porc_stp5 = SVGMobject('fig_porcine_stp5.svg').scale(scale).shift(shift)
        porc_stp6 = SVGMobject('fig_porcine_stp6.svg').scale(scale).shift(shift)
        self.play(Write(porc_stp5))
        self.play(Write(porc_stp6))
        self.slide_break()

        self.play(FadeOut(porc_stp1, porc_stp2, porc_stp3, porc_stp4, porc_stp5, porc_stp6, nnmat_title, nnmat_cite))
        self.slide_break()


        """
        End of NNMAT paper
        """



        """
        NODE paper
        """

        datadriven = Tex(" ", "Data-driven", " strain energy density functions")
        consistent = Tex(" ", "Thermodynamically consistent", " strain energy density functions")
        objective = Tex(" ", "Objective", " strain energy density functions")
        polyconvex = Tex(" ", "Polyconvex", " strain energy density functions")

        for obj in [datadriven, consistent, objective, polyconvex]:
            obj[1].set_color(YELLOW)
        
        
        self.play(FadeIn(datadriven))
        self.wait()
        self.slide_break()

        self.play(ReplacementTransform(datadriven, consistent))
        self.wait()
        self.play(ReplacementTransform(consistent, objective))
        self.wait()
        self.play(ReplacementTransform(objective, polyconvex))

        self.slide_break()
        whatispoly = Tex("What is ", "Polyconvex", "ity?")
        whatispoly[1].set_color(YELLOW)
        whatispoly[2].set_color(YELLOW)
        self.play(ReplacementTransform(polyconvex, whatispoly))
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

        self.remove(Psi)
        Psi3 = MathTex("\Psi", " = \Psi_{I_1}(", "I_1", ") + \Psi_{I_2}(", "I_2", ") + \Psi_{I_3}(", "I_3", ") + \Psi_{I_{4v}}(", "I_{4v}", ") + ...")
        self.play(ReplacementTransform(Psi[0],Psi3[0]), ReplacementTransform(Psi[1:], Psi3[1:]))
        self.slide_break()
        
        self.remove(Psi2)


        Psi4 = MathTex("\Psi", " = ", "\Psi_{I_1}", "(", "I_1", ") + ", "\Psi_{I_2}", "(", "I_2", ") + ", "\Psi_{I_3}", 
                        "(", "I_3", ") + ", "\Psi_{I_{4v}}", "(", "I_{4v}", ") + ...")
        self.remove(Psi3)

        self.play(*[Circumscribe(Psi4[i], run_time=1.7) for i in [2, 6, 10, 14]])
        self.slide_break()


        Psi5 = MathTex("\Psi", " = ", "\Psi_{I_1}(I_1)", " + ", "\Psi_{I_2}(I_2)", " + ", "\Psi_{I_3}(I_3)", " + ", 
                    "\Psi_{I_{4v}}(I_{4v})", " + ...")
        # self.play(FadeIn(Psi2))
        self.add(Psi5)
        self.play(FadeOut(Psi, Psi3, Psi4), run_time=0.01) # I want to remove artifacts, but self.remove doesn't work for some reason.

        dPsi1 = MathTex("\\frac{\partial \Psi}{\partial \mathbf{C}}", " = ", "\Psi_{I_1}(I_1)", " + ", "\Psi_{I_2}(I_2)", " + ", "\Psi_{I_3}(I_3)", " + ", 
                    "\Psi_{I_{4v}}(I_{4v})", " + ...").align_to(Psi5[0], LEFT+DOWN)
        dPsi2 = MathTex("\\frac{\partial \Psi}{\partial \mathbf{C}}", " = ", 
                        "\\frac{\partial \Psi_{I_1}}{\partial I_1}\\frac{\partial I_1}{\partial \mathbf{C}}", 
                        " + ", "\Psi_{I_2}(I_2)", " + ", "\Psi_{I_3}(I_3)", " + ", 
                        "\Psi_{I_{4v}}(I_{4v})", " + ...").align_to(Psi5[0], LEFT+DOWN)
        dPsi3 = MathTex("\\frac{\partial \Psi}{\partial \mathbf{C}}", " = ", 
                        "\\frac{\partial \Psi_{I_1}}{\partial I_1}\\frac{\partial I_1}{\partial \mathbf{C}}", 
                        " + ", "\\frac{\partial \Psi_{I_2}}{\partial I_2}\\frac{\partial I_2}{\partial \mathbf{C}}", 
                        " + ", "\Psi_{I_3}(I_3)", " + ", 
                        "\Psi_{I_{4v}}(I_{4v})", " + ...").align_to(Psi5[0], LEFT+DOWN)
        dPsi4 = MathTex("\\frac{\partial \Psi}{\partial \mathbf{C}}", " = ", 
                        "\\frac{\partial \Psi_{I_1}}{\partial I_1}\\frac{\partial I_1}{\partial \mathbf{C}}", 
                        " + ", "\\frac{\partial \Psi_{I_2}}{\partial I_2}\\frac{\partial I_2}{\partial \mathbf{C}}", 
                        " + ", "\\frac{\partial \Psi_{I_3}}{\partial I_3}\\frac{\partial I_3}{\partial \mathbf{C}}", " + ", 
                        "\Psi_{I_{4v}}(I_{4v})", " + ...").align_to(Psi5[0], LEFT+DOWN)
        dPsi5 = MathTex("\\frac{\partial \Psi}{\partial \mathbf{C}}", " = ", 
                        "\\frac{\partial \Psi_{I_1}}{\partial I_1}\\frac{\partial I_1}{\partial \mathbf{C}}", 
                        " + ", "\\frac{\partial \Psi_{I_2}}{\partial I_2}\\frac{\partial I_2}{\partial \mathbf{C}}", 
                        " + ", "\\frac{\partial \Psi_{I_3}}{\partial I_3}\\frac{\partial I_3}{\partial \mathbf{C}}", " + ...").align_to(Psi5[0], LEFT+DOWN)

        self.play(ReplacementTransform(Psi5, dPsi1))
        self.add(dPsi1)
        self.remove(*Psi5[0])
        self.play(ReplacementTransform(dPsi1, dPsi2))
        self.add(dPsi2)
        self.remove(dPsi1)
        self.play(ReplacementTransform(dPsi2, dPsi3))
        self.add(dPsi3)
        self.remove(dPsi2)
        self.play(ReplacementTransform(dPsi3, dPsi4))
        self.add(dPsi4)
        self.remove(dPsi3)
        self.play(ReplacementTransform(dPsi4[7:], dPsi5[7]))
        self.add(dPsi5)
        self.remove(dPsi4)
        self.remove(Psi5[1])

        self.remove(*dPsi1, *dPsi2, *dPsi3, *dPsi4, *Psi2[0], *Psi2[1])
        self.slide_break()


        partials = MathTex("\\frac{\partial \Psi_{I_i}}{\partial I_i} \quad \\rightarrow").move_to(1.5*DOWN + LEFT)
        conditions = Group(
            Tex("Monotonic"),
            Tex("Non-negative")
        ).arrange(DOWN, buff=0.5).next_to(partials, RIGHT).shift(0.7*RIGHT )
        conditions[0].set_color(GREEN)
        conditions[1].set_color(BLUE)
        self.play(Write(partials))
        self.play(Write(conditions[0]))
        self.play(Write(conditions[1]))
        self.slide_break()

        self.play(FadeOut(*dPsi5, partials, conditions[0], conditions[1]))


class hyper_p2_node(SlideMovingCameraScene):
    def construct(self):
        heading = toc[0].copy().move_to(ORIGIN).scale(1.25).to_corner(UP)
        self.add(heading)

        node1 = Tex("Neural ODE").set_color(YELLOW)
        node2 = MathTex("\mathcal{N}").set_color(YELLOW).scale(2)

        self.play(Write(node1))
        self.slide_break()
        self.play(Transform(node1, node2))

        nodebox = Square(side_length=2.0)
        inparr = Arrow([-2.5,0,0], [-0.8,0,0])
        outarr = Arrow([+0.8,0,0], [+2.5,0,0])
        inptext = Tex("Input").move_to([-3.3,0,0])
        outtext = Tex("Output").move_to([+3.3,0,0])
        self.play(Create(nodebox))
        self.play(FadeIn(inptext))
        self.play(Create(inparr))
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
        self.play(Create(axis1), run_time=0.5)
        self.play(FadeIn(h), run_time=0.5)
        self.play(FadeIn(t), run_time=0.5)
        self.play(Write(h0), run_time=0.5)
        self.play(FadeIn(inp), run_time=0.5)
        self.play(Write(h1), run_time=0.5)
        self.play(Create(dashedline), run_time=0.5)
        self.play(FadeIn(out), run_time=0.5)
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

        self.slide_break()
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
        self.play(*[item.animate.shift(LEFT*2.8) for item in [axis1, t, h, h0, h1, dashedline, inp, out]], run_time=0.5)

        #Input-output map
        axis2 = Axes([0,0.6], [0.1,1.25], x_length=8, axis_config={"include_ticks":False}).scale(0.5).shift(RIGHT*2.8)
        inp2 = Tex("Input").move_to(axis2.coords_to_point(0.6,-0.12))
        out2 = Tex("Output").move_to(axis2.coords_to_point(0,1.4))
        self.play(Create(axis2), run_time=0.2)
        self.play(FadeIn(inp2), run_time=0.2)
        self.play(FadeIn(out2), run_time=0.2)

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
        self.play(Create(dot1), run_time=0.2)
        self.play(Create(graph), run_time=0.2)
        self.play(Create(dot2), Create(dot), run_time=0.2)

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
            self.play(Create(dot1), run_time=0.2)
            self.play(Create(graph), run_time=0.2)
            self.play(Create(dot), Create(dot2), run_time=0.2)
            self.play(Create(graph2), run_time=0.2)

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
        self.play(Create(dot1), run_time=0.5)
        self.play(Flash(dot1), run_time=0.5)
        self.play(Create(graph), run_time=0.5)
        self.play(Create(dot2), Create(dot), run_time=0.5)
        self.play(Flash(dot2), run_time=0.5)
        self.play(Create(graph2), run_time=0.5)
        self.play(Flash(dot), run_time=0.5)
        self.play(Write(nonnegative), run_time=0.5)
        self.slide_break()


        self.play(*[FadeOut(obj) for obj in self.mobjects])

        self.camera.frame.save_state()
        self.play(self.camera.frame.animate.set(width=1.8), run_time=0.2)
        self.add(heading, nodebox, inptext, outtext, inparr, outarr)
        self.play(Restore(self.camera.frame), FadeIn(node1), run_time=1.5)

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
        self.wait(0.1)
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
        self.wait(0.1)
        self.slide_break()

        self.play(FadeOut(node1, nodebox, inptext, outtext, inparr, outarr, dPsi4))
        self.wait(0.1)
        # self.play(dPsi4.animate.move_to(ORIGIN))
        # self.remove(*[obj for obj in self.mobjects])
        # self.add(heading)


class hyper_p3_convexproof(SlideScene):
    def construct(self):
        heading = toc[0].copy().move_to(ORIGIN).scale(1.25).to_corner(UP)
        self.add(heading)

        node_title = Tex("\\begin{tabular}{c}Data-driven tissue mechanics with polyconvex neural ordinary differential equations\\textsuperscript{2}\\end{tabular}").scale(0.7).move_to(2.5*UP)
        node_cite = Tex("\\begin{tabular}{c}\\textsuperscript{2}V. Tac, F. Sahli Costabal, A. Buganza Tepole, Comput Method App Mech Eng, 2022.\\end{tabular}").scale(0.5).move_to(3.5*DOWN)
        self.play(Write(node_title))
        self.play(FadeIn(node_cite))
        self.slide_break()

        scale = 2
        GOH_gt = SVGMobject("convexity_GOH_gt.svg").scale(scale).shift(DOWN+0.6*LEFT)
        GOH_pr = SVGMobject("convexity_GOH_pr.svg").scale(scale).move_to(GOH_gt).shift(3.5*RIGHT)
        Fung_gt = SVGMobject("convexity_Fung_gt.svg").scale(scale)
        Fung_pr = SVGMobject("convexity_Fung_pr.svg").scale(scale)
        txt1 = SVGMobject("convexity_text.svg").scale(scale).move_to(GOH_gt)
        txt2 = txt1.copy().move_to(GOH_pr)
        label1 = Tex("GOH data").next_to(GOH_gt, UP).shift(0.6*RIGHT)
        label2 = Tex("NODE predictions").next_to(GOH_pr, UP).shift(0.6*RIGHT)
        self.play(FadeIn(label1))
        self.play(Write(GOH_gt), Write(txt1), run_time=3)
        self.slide_break()
        self.play(GOH_gt.animate.shift(3*LEFT), txt1.animate.shift(3*LEFT), label1.animate.shift(3*LEFT))
        self.play(Write(GOH_pr), Write(txt2), FadeIn(label2))
        self.slide_break()


        conv_outline1 = SVGMobject("convexity_outline1.svg").scale(scale).move_to(GOH_pr)
        conv_outline2 = SVGMobject("convexity_outline2.svg").scale(scale).move_to(GOH_pr)
        conv_outline3 = SVGMobject("convexity_outline3.svg").scale(scale).move_to(GOH_pr)
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


class hyper_p4_murine(SlideThreeDScene):
    def construct(self):
        heading = toc[0].copy().move_to(ORIGIN).scale(1.25).to_corner(UP)
        self.add(heading)
        node_title = Tex("\\begin{tabular}{c}Data-driven tissue mechanics with polyconvex neural ordinary differential equations\\textsuperscript{2}\\end{tabular}").scale(0.7).move_to(2.5*UP)
        node_cite = Tex("\\begin{tabular}{c}\\textsuperscript{2}V. Tac, F. Sahli Costabal, A. Buganza Tepole, Comput Method App Mech Eng, 2022.\\end{tabular}").scale(0.5).move_to(3.5*DOWN)
        self.add(node_title, node_cite)

        cbar = ImageMobject('cbar.png').move_to(ORIGIN).move_to(6*LEFT).scale(2)
        cbar_label = Tex(r'Error \\\ [MPa]').move_to(ORIGIN).move_to(5.2*LEFT).scale(0.7)
        cbar_tick1 = Tex("0.0").align_to(cbar_label, LEFT).shift(1.7*DOWN).scale(0.7)
        cbar_tick2 = Tex("0.15").align_to(cbar_label, LEFT).shift(1.7*UP).scale(0.7)
        self.add_fixed_in_frame_mobjects(heading, cbar, cbar_label, cbar_tick1, cbar_tick2, node_title, node_cite)
        self.remove(cbar, cbar_label, cbar_tick1, cbar_tick2)

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
            ).scale(0.3).move_to(ORIGIN).shift(0.5*DOWN)
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
                self.play(FadeIn(dot), run_time=0.01)
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


class hyper_p5_boxplots(SlideScene):
    def construct(self):
        heading = toc[0].copy().move_to(ORIGIN).scale(1.25).to_corner(UP)
        self.add(heading)
        node_title = Tex("\\begin{tabular}{c}Data-driven tissue mechanics with polyconvex neural ordinary differential equations\\textsuperscript{2}\\end{tabular}").scale(0.7).move_to(2.5*UP)
        node_cite = Tex("\\begin{tabular}{c}\\textsuperscript{2}V. Tac, F. Sahli Costabal, A. Buganza Tepole, Comput Method App Mech Eng, 2022.\\end{tabular}").scale(0.5).move_to(3.5*DOWN)
        self.add(node_title, node_cite)

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

        self.play(*[FadeIn(label) for label in labels], run_time=0.5)
        self.play(FadeIn(trn_legend_bar), FadeIn(trn_legend_tex), run_time=0.5)
        self.play(*[Create(item) for item in trn_median_bars], run_time=0.5)
        self.play(*[GrowFromEdge(item, DOWN) for item in trn_boxes_upper], *[GrowFromEdge(item, UP) for item in trn_boxes_lower], run_time=0.5)
        self.play(*[Create(item) for item in trn_wh_body_upper], *[Create(item) for item in trn_wh_body_lower], run_time=0.5)
        self.play(*[Create(item) for item in trn_wh_ends_upper], *[Create(item) for item in trn_wh_ends_lower], run_time=0.5)

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

        self.play(FadeIn(val_legend_bar), FadeIn(val_legend_tex), run_time=0.5)
        self.play(*[Create(item) for item in val_median_bars], run_time=0.5)
        self.play(*[GrowFromEdge(item, DOWN) for item in val_boxes_upper], *[GrowFromEdge(item, UP) for item in val_boxes_lower], run_time=0.5)
        self.play(*[Create(item) for item in val_wh_body_upper], *[Create(item) for item in val_wh_body_lower], run_time=0.5)
        self.play(*[Create(item) for item in val_wh_ends_upper], *[Create(item) for item in val_wh_ends_lower], run_time=0.5)
        self.slide_break()


        # Remove everything
        self.play(*[FadeOut(obj) for obj in [axes, *labels, ylabel, *trn_legend_bar, *trn_legend_tex, *trn_median_bars, 
                                            *trn_boxes_upper, *trn_boxes_lower, *trn_wh_body_upper, *trn_wh_body_lower,
                                            *trn_wh_ends_upper, *trn_wh_ends_lower, *val_legend_bar, *val_legend_tex, *val_median_bars, 
                                            *val_boxes_upper, *val_boxes_lower, *val_wh_body_upper, *val_wh_body_lower,
                                            *val_wh_ends_upper, *val_wh_ends_lower, heading, node_title, node_cite]])
        # self.play(FadeOut(node_title, node_cite))


class hyper_p6_benchmark(SlideScene):
    def construct(self):
        heading = toc[0].copy().move_to(ORIGIN).scale(1.25).to_corner(UP)
        self.add(heading)

        bench_title = Tex("\\begin{tabular}{c}Benchmarking physics-informed frameworks for data-driven hyperelasticity\\textsuperscript{3}\\end{tabular}").scale(0.7).move_to(2.5*UP)
        bench_cite = Tex("\\begin{tabular}{c}\\textsuperscript{3}V. Tac, K. Linka, F. Sahli Costabal, E. Kuhl, A. Buganza Tepole, Computational Mechanics, 2024.\\end{tabular}").scale(0.5).move_to(3.5*DOWN)
        self.play(Write(bench_title))
        self.play(FadeIn(bench_cite))


        node_diag = SVGMobject('node_diag.svg').scale(2).set_color(WHITE).shift(0.5*DOWN)
        self.play(FadeIn(node_diag))
        self.slide_break()

        cann_diag = SVGMobject('cann_diag.svg').scale(2).set_color(WHITE).shift(0.5*DOWN).shift(15*RIGHT)
        self.play(node_diag.animate.shift(15*LEFT), cann_diag.animate.shift(15*LEFT))
        self.slide_break()

        icnn_diag = SVGMobject('icnn_diag.svg').scale(2).set_color(WHITE).shift(0.5*DOWN).shift(15*RIGHT)
        self.play(cann_diag.animate.shift(15*LEFT), icnn_diag.animate.shift(15*LEFT))
        self.slide_break()

        self.play(icnn_diag.animate.shift(15*LEFT))

        self.play(FadeOut(heading, bench_title, bench_cite))
        scale = 4
        rubber_stp1 = SVGMobject('fig_rubber_stp1.svg').scale(scale).set_color(WHITE)
        self.play(Create(rubber_stp1))
        self.slide_break()
        rubber_stp2 = SVGMobject('fig_rubber_stp2.svg').scale(scale).set_color(WHITE)
        self.play(Create(rubber_stp2))
        self.slide_break()
        rubber_stp3 = SVGMobject('fig_rubber_stp3.svg').scale(scale).set_color(WHITE)
        self.play(Create(rubber_stp3))
        self.slide_break()
        rubber_stp4 = SVGMobject('fig_rubber_stp4.svg').scale(scale).set_color(WHITE)
        self.play(Create(rubber_stp4))
        self.slide_break()
        rubber_stp5 = SVGMobject('fig_rubber_stp5.svg').scale(scale).set_color(WHITE)
        self.play(Create(rubber_stp5))
        self.slide_break()
        rubber_stp6 = SVGMobject('fig_rubber_stp6.svg').scale(scale).set_color(WHITE)
        self.play(Create(rubber_stp6))
        self.slide_break()
        rubber_stp7 = SVGMobject('fig_rubber_stp7.svg').scale(scale).set_color(WHITE)
        self.play(Create(rubber_stp7))
        self.slide_break()
        rubber_stp8 = SVGMobject('fig_rubber_stp8.svg').scale(scale).set_color(WHITE)
        self.play(Create(rubber_stp8))
        self.slide_break()
        rubber_stp9 = SVGMobject('fig_rubber_stp9.svg').scale(scale).set_color(WHITE)
        self.play(Create(rubber_stp9))
        self.slide_break()
        rubber_stp10 = SVGMobject('fig_rubber_stp10.svg').scale(scale).set_color(WHITE)
        self.play(Create(rubber_stp10))
        self.slide_break()
        rubber_stp11 = SVGMobject('fig_rubber_stp11.svg').scale(scale).set_color(WHITE)
        self.play(Create(rubber_stp11))
        self.slide_break()
        rubber_stp12 = SVGMobject('fig_rubber_stp12.svg').scale(scale).set_color(WHITE)
        self.play(Create(rubber_stp12))
        self.slide_break()

        self.play(FadeOut(rubber_stp1, rubber_stp2, rubber_stp3, rubber_stp4, 
                          rubber_stp5, rubber_stp6, rubber_stp7, rubber_stp8, 
                          rubber_stp9, rubber_stp10, rubber_stp11, rubber_stp12))


class visco(SlideScene):
    def construct(self):
        self.add(toc, toctitle)

        heading = toc[1].copy().move_to(ORIGIN).scale(1.25).to_corner(UP)
        self.play(FadeOut(toc, toctitle), ReplacementTransform(toc[1],heading))
        self.slide_break()

        self.play(heading[1].animate.scale(1.2).set_color(YELLOW))
        self.play(heading[1].animate.scale(1/1.2).set_color(WHITE))
        self.slide_break()

        """
        Start (almost direct) copy from the nvisco video
        """
        # self.wait(0.5)
        heart = SVGMobject("visco_figs/heart.svg").scale(1.2).move_to(5*LEFT + 1*UP)
        hearttitle = Tex("Myocardium").move_to(5*LEFT+2.5*UP).scale(0.7)
        self.play(Write(heart), FadeIn(hearttitle))
        brain = SVGMobject("visco_figs/brain.svg").move_to(1.8*LEFT + 1*UP)
        braintitle = Tex("Brain tissue").move_to(1.8*LEFT+2.5*UP).scale(0.7)
        self.play(Write(brain), FadeIn(braintitle))
        wheel = SVGMobject("visco_figs/wheel.svg").move_to(1.8*RIGHT + 1*UP)
        wheeltitle = Tex("Rubber").move_to(1.8*RIGHT+2.5*UP).scale(0.7)
        self.play(Write(wheel), FadeIn(wheeltitle))
        bc = SVGMobject("visco_figs/bc.svg").move_to(5 *RIGHT + 1*UP)
        bctitle = Tex("Blood clots").move_to(5*RIGHT+2.5*UP).scale(0.7)
        self.play(Write(bc), FadeIn(bctitle))
        self.slide_break()

        requirements = Group(
            Tex('Modeling viscoelasticity:'),
            Tex('1. Flexible models - data-driven methods'),
            Tex('2. Rigorous formulation'),
            Tex('3. Physical constraints - 2\\textsuperscript{nd} Law of Thermodynamics, etc.'),
            Tex('4. Scope (large and rapid deformations) - Finite viscoelasticity'),
            Tex('5. Anisotropic behavior')
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).scale(0.6).move_to(2*DOWN+1.5*LEFT)

        for req in requirements:
            self.play(Write(req))
            self.slide_break()

        sample = Rectangle(width=0.5, height=0.5).move_to(heart).shift(0.3*RIGHT+0.5*DOWN).set_color(YELLOW)
        FD = SVGMobject("visco_figs/FD.svg").move_to(sample).scale(0.2)
        self.play(Create(sample))

        self.play(sample.animate.move_to(5*RIGHT+DOWN).scale(3),
                  FadeIn(FD),
                  FD.animate.move_to(5*RIGHT+DOWN).scale(3.4),
                  FadeOut(heart, hearttitle, brain, braintitle, wheel, wheeltitle, bc, bctitle, requirements))

        MFDarrow = Arrow([0,0,0], [0,1,0]).scale(2.5).next_to(sample,LEFT)
        MFD = Tex("Mean Fiber Direction").next_to(MFDarrow,LEFT).scale(0.7).shift(0.8*RIGHT)
        self.play(Create(MFDarrow), FadeIn(MFD))
        self.play(FadeOut(MFD, MFDarrow))

        CFDarrow = Arrow([0,0,0], [1,0,0]).scale(2.5).next_to(sample,UP)
        CFD = Tex("Cross Fiber Direction").next_to(CFDarrow,UP).shift(0.2*LEFT).scale(0.7)
        self.play(Create(CFDarrow), FadeIn(CFD))
        self.play(FadeOut(CFD, CFDarrow, FD)) 
        self.slide_break()

        MFDaxes = Axes(
            [0,310], 
            [0,17],
            x_length=6,
            y_axis_config={'include_ticks':False},
            x_axis_config={'include_ticks':False}
            ).scale(0.5).move_to(4*LEFT+DOWN)
        MFDylabel = Tex("Stress").scale(0.7).rotate(90*DEGREES).next_to(MFDaxes,LEFT)
        MFDxlabel = Tex("Time").scale(0.7).next_to(MFDaxes,DOWN)
        MFDtitle = Tex("Mean fiber direction").scale(0.7).next_to(MFDaxes,UP)
         
        CFDaxes = Axes(
            [0,310], 
            [0,17],
            x_length=6,
            y_axis_config={'include_ticks':False},
            x_axis_config={'include_ticks':False}
            ).scale(0.5).move_to(1*RIGHT+DOWN)
        CFDylabel = Tex("Stress").scale(0.7).rotate(90*DEGREES).next_to(CFDaxes,LEFT)
        CFDxlabel = Tex("Time").scale(0.7).next_to(CFDaxes,DOWN)
        CFDtitle = Tex("Cross fiber direction").scale(0.7).next_to(CFDaxes,UP)

        MFDdata = np.genfromtxt("fig_myocardium1_a.csv")[1:]
        CFDdata = np.genfromtxt("fig_myocardium1_b.csv")[1:]
        time = MFDdata[:,0]
        MFDstress_gt = MFDdata[:,1]
        MFDstress_nn = MFDdata[:,2]
        CFDstress_gt = CFDdata[:,1]
        CFDstress_nn = CFDdata[:,2]

        MFDpoints = []
        CFDpoints = []
        for t, MFDp, CFDp in zip(time, MFDstress_gt, CFDstress_gt):
            MFDpoints.append(Dot(MFDaxes.coords_to_point(t+5,MFDp+0.5,0)).scale(0.4).set_color(YELLOW))
            CFDpoints.append(Dot(CFDaxes.coords_to_point(t+5,CFDp+0.5,0)).scale(0.4).set_color(YELLOW))

        boundup = Rectangle(width=2, height=0.2).move_to(sample).shift(0.87*UP)
        bounddown = Rectangle(width=2, height=0.2).move_to(sample).shift(0.87*DOWN)
        boundleft = Rectangle(width=0.2, height=2).move_to(sample).shift(0.87*LEFT)
        boundright = Rectangle(width=0.2, height=2).move_to(sample).shift(0.87*RIGHT)
        arrowup = Arrow([0,0,0],[0,1,0]).next_to(boundup, UP).shift(0.2*DOWN)
        arrowdown = Arrow([0,1,0],[0,0,0]).next_to(bounddown, DOWN).shift(0.2*UP)
        arrowleft = Arrow([1,0,0],[0,0,0]).next_to(boundleft, LEFT).shift(0.2*RIGHT)
        arrowright = Arrow([0,0,0],[1,0,0]).next_to(boundright, RIGHT).shift(0.2*LEFT)
        arrowup_label = Tex("Stress").scale(0.5).next_to(arrowup, LEFT)
        arrowleft_label = Tex("Stress").scale(0.5).next_to(arrowleft, UP).shift(0.2*LEFT)

        arrowup_node = arrowup.copy()
        arrowdown_node = arrowdown.copy()
        arrowleft_node = arrowleft.copy()
        arrowright_node = arrowright.copy()
        arrowup_label_node = arrowup_label.copy()
        arrowleft_label_node = arrowleft_label.copy()

        boundup.set_fill(WHITE, opacity=1.0)
        bounddown.set_fill(WHITE, opacity=1.0)
        boundleft.set_fill(WHITE, opacity=1.0)
        boundright.set_fill(WHITE, opacity=1.0)
        experiments = Tex("Experiments show history dependent behavior").scale(0.75).shift(2.2*UP+2.5*LEFT)
        experiments[0][:11].set_color(YELLOW)
        self.play(Create(MFDaxes), Create(CFDaxes), Create(experiments), FadeIn(MFDylabel, MFDxlabel, CFDylabel, CFDxlabel, MFDtitle, CFDtitle))

        sample2 = sample.copy()
        sample3 = sample.copy()
        sample_node = sample.copy()
        sample_tall = Rectangle(height=2.3, width=1.5).set_color(YELLOW).move_to(sample)
        self.add(arrowup, arrowdown, arrowup_label)
        self.play(boundup.animate.shift(0.4*UP), 
                  bounddown.animate.shift(0.4*DOWN), 
                  Transform(sample, sample_tall), 
                  FadeIn(*MFDpoints[:12]),
                  arrowup.animate.shift(0.4*UP), 
                  arrowdown.animate.shift(0.4*DOWN),
                  arrowup_label.animate.shift(0.4*UP))
        for p in MFDpoints[12::2]:
            self.play(FadeIn(p), arrowup.animate.scale(0.97), arrowdown.animate.scale(0.97), run_time=0.05)
        self.remove(sample)
        self.add(sample_tall)

        self.play(FadeOut(boundup, bounddown, arrowup, arrowdown, arrowup_label), run_time=0.4)
        self.play(Transform(sample_tall, sample2), run_time=0.4)
        self.remove(sample_tall)
        self.add(sample2)

        self.play(FadeIn(boundright, boundleft))
        sample_wide = Rectangle(height=1.5, width=2.3).set_color(YELLOW).move_to(sample)
        self.add(arrowleft, arrowright, arrowleft_label)
        self.play(boundleft.animate.shift(0.4*LEFT), 
                  boundright.animate.shift(0.4*RIGHT), 
                  Transform(sample2, sample_wide), 
                  FadeIn(*CFDpoints[:12]),
                  arrowleft.animate.shift(0.4*LEFT), 
                  arrowright.animate.shift(0.4*RIGHT),
                  arrowleft_label.animate.shift(0.4*LEFT))
        for p in CFDpoints[12::2]:
            self.play(FadeIn(p), arrowleft.animate.scale(0.97), arrowright.animate.scale(0.97), run_time=0.05)
        self.remove(sample2)
        self.add(sample_wide)

        self.play(FadeOut(boundleft, boundright, arrowleft, arrowright, arrowleft_label), run_time=0.4)
        self.play(Transform(sample_wide, sample3), run_time=0.4)
        self.remove(sample_wide)
        self.add(sample3)

        self.slide_break()
        self.play(FadeOut(experiments))
 

        #NODE
        nodetitle = Tex("Fully data-driven models of finite viscoelasticity with Neural ODEs\\textsuperscript{4}").scale(0.75).move_to(experiments, aligned_edge=LEFT)
        nvisco_cite = Tex("\\begin{tabular}{c}\\textsuperscript{4}V. Tac, M.K. Rausch, F. Sahli Costabal, A. Buganza Tepole, CMAME, 2023.\\end{tabular}").scale(0.5).move_to(3.5*DOWN)
        nodetitle[0][-11:-1].set_color(BLUE_C)

        sample_node.set_color(BLUE_C)
        sample_node2 = sample_node.copy()
        sample_node3 = sample_node.copy()
        sample_node_tall = Rectangle(height=2.3, width=1.5).set_color(BLUE_C).move_to(sample_node)
        sample_node_wide = Rectangle(height=1.5, width=2.3).set_color(BLUE_C).move_to(sample_node)

        sample_node.shift(3*RIGHT)
        self.play(Write(nodetitle), sample3.animate.shift(5*RIGHT), FadeIn(nvisco_cite))
        self.play(sample_node.animate.shift(3*LEFT))

        boundup.shift(0.4*DOWN) #Restore to their original positions
        bounddown.shift(0.4*UP)
        boundright.shift(0.4*LEFT)
        boundleft.shift(0.4*RIGHT)
        self.play(FadeIn(boundup, bounddown))
        MFDgraph1 = MFDaxes.plot_line_graph(time[:12]+5, MFDstress_nn[:12]+0.5, add_vertex_dots=False, stroke_width=7).set_color(BLUE_C)
        self.add(arrowup_node, arrowdown_node, arrowup_label_node)
        self.play(boundup.animate.shift(0.4*UP), 
                  bounddown.animate.shift(0.4*DOWN), 
                  Transform(sample_node, sample_node_tall), 
                  Create(MFDgraph1),
                  arrowup_node.animate.shift(0.4*UP), 
                  arrowdown_node.animate.shift(0.4*DOWN),
                  arrowup_label_node.animate.shift(0.4*UP))
        self.remove(sample_node)
        self.add(sample_node_tall)
        MFDgraph2 = MFDaxes.plot_line_graph(time[12:]+5, MFDstress_nn[12:]+0.5, add_vertex_dots=False, stroke_width=7).set_color(BLUE_C)
        self.play(Create(MFDgraph2), arrowup_node.animate.scale(0.3), arrowdown_node.animate.scale(0.3), run_time=3, rate_func=rate_functions.linear)

        self.play(FadeOut(boundup, bounddown, arrowup_node, arrowdown_node, arrowup_label_node), run_time=0.4)
        self.play(Transform(sample_node_tall, sample_node2), run_time=0.4)
        self.remove(sample_node_tall)
        self.add(sample_node2)


        self.play(FadeIn(boundright, boundleft))
        CFDgraph1 = CFDaxes.plot_line_graph(time[:12]+5, CFDstress_nn[:12]+0.5, add_vertex_dots=False, stroke_width=7).set_color(BLUE_C)
        self.add(arrowleft_node, arrowright_node, arrowleft_label_node)
        self.play(boundright.animate.shift(0.4*RIGHT), 
                  boundleft.animate.shift(0.4*LEFT), 
                  Transform(sample_node2, sample_node_wide), 
                  Create(CFDgraph1),
                  arrowleft_node.animate.shift(0.4*LEFT), 
                  arrowright_node.animate.shift(0.4*RIGHT),
                  arrowleft_label_node.animate.shift(0.4*LEFT))
        self.remove(sample_node2)
        self.add(sample_node_wide)
        CFDgraph2 = CFDaxes.plot_line_graph(time[12:]+5, CFDstress_nn[12:]+0.5, add_vertex_dots=False, stroke_width=7).set_color(BLUE_C)
        self.play(Create(CFDgraph2), arrowleft_node.animate.scale(0.3), arrowright_node.animate.scale(0.3), run_time=3, rate_func=rate_functions.linear)

        self.play(FadeOut(boundright, boundleft, arrowleft_node, arrowright_node, arrowleft_label_node), run_time=0.4)
        self.play(Transform(sample_node_wide, sample_node3), run_time=0.4)
        self.remove(sample_node_wide)
        self.add(sample_node3)
        self.slide_break()
        
        self.play(FadeOut(sample_node3, CFDtitle, MFDtitle, CFDxlabel, CFDylabel, MFDxlabel, 
                          MFDylabel, CFDaxes, MFDaxes, CFDgraph1, CFDgraph2, MFDgraph1, MFDgraph2,
                          *MFDpoints[:12], *MFDpoints[12::2], *CFDpoints[:12], *CFDpoints[12::2]))

        """
        End (almost direct) copy from the nvisco video
        """
        
        divider1 = Line([-2.6,-1,0], [-2.6,1,0])
        divider2 = Line([1.7,-1,0], [1.7,1,0])
        self.play(Create(divider1), Create(divider2))

        eq = Tex("Equilibrium").shift(4.3*LEFT)
        psi_eq = MathTex("\Psi_{EQ}").move_to(eq)
        neq = Tex("Non-equilibrium").shift(0.3*LEFT)
        psi_neq = MathTex("\Psi_{NEQ}").move_to(neq)
        diss = Tex("Dissipation potential").shift(4.2*RIGHT)
        phi = MathTex("\Phi").move_to(diss)


        self.play(Write(eq), Write(neq), Write(diss))
        self.slide_break()

        self.play(ReplacementTransform(eq, psi_eq))
        self.slide_break()
        self.play(ReplacementTransform(neq, psi_neq))
        self.slide_break()
        self.play(ReplacementTransform(diss, phi))
        self.slide_break()

 

        self.play(*[obj.animate.scale(1.2).set_color(YELLOW)  for obj in [psi_eq, psi_neq]])
        self.play(*[obj.animate.scale(1/1.2).set_color(WHITE) for obj in [psi_eq, psi_neq]])
        self.slide_break()

        self.play(phi.animate.scale(1.2).set_color(YELLOW))
        self.play(phi.animate.scale(1/1.2))
        self.slide_break()

        # self.play(*[FadeOut(obj) for obj in [psi_eq, psi_neq, divider1]])
        self.play(*[obj.animate.shift(20*LEFT) for obj in [eq, psi_eq, neq, psi_neq, divider1, divider2]], phi.animate.shift(4.2*LEFT))

        self.play(phi.animate.shift(1.2*UP))

        evol_eq = MathTex(r"\mathcal{L}\mathbf{b}_e\mathbf{b}_e^{-1} = ", r"\frac{\partial \Phi}{\partial \tau_{NEQ}}")
        timestau = MathTex(r": \tau_{NEQ}", "\ge 0").next_to(evol_eq, RIGHT)
        self.play(Write(evol_eq))
        self.slide_break() 

        self.play(Unwrite(evol_eq[0]))
        self.play(Write(timestau))
        self.play(evol_eq.animate.shift(LEFT*2.5), timestau.animate.shift(LEFT*2.5))
        self.slide_break()

        positivediss = Group(
            Tex('Positive dissipation criterion (2\\textsuperscript{nd} Law):'),
            Tex('$\\bullet \,\,\, \Phi$ : Convex in $\\tau_{NEQ}, \quad$ or'),
            Tex('$\\bullet \,\,\, \partial \Phi / \partial \\tau_{NEQ}$ : Monotonic in $\\tau_{NEQ}$'),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).scale(0.7).move_to(3*LEFT+2*DOWN)

        for item in positivediss:
            self.play(Write(item))
            self.slide_break()

        self.play(FadeOut(evol_eq[1]), FadeOut(timestau), FadeOut(nvisco_cite, phi, nodetitle, positivediss))

        scale = 2
        shift = 0.5*DOWN
        diss_stp1 = SVGMobject('visco_figs/diss_toprow_stp1.svg').scale(scale).shift(shift)
        diss_label = Tex('Dissipation').rotate(90*DEGREES).scale(0.7).move_to(5.5*LEFT+0.2*DOWN)
        labelset1 = MathTex('-1 \,\,\, (\\tau_{NEQ})_{11} \,\,\, 1').scale(0.7).move_to(3.8*LEFT+2*DOWN)
        labelset2 = labelset1.copy().shift(4*RIGHT)
        labelset3 = labelset2.copy().shift(4.25*RIGHT )
        self.play(Write(diss_stp1), Write(diss_label), Write(labelset1))
        self.slide_break()

        uni = Tex('Uniaxial').scale(0.7).move_to(3.8*LEFT+1*UP)
        equ = Tex('Equibiaxial').scale(0.7).move_to(0.25*RIGHT+1*UP)
        ps = Tex('Pure shear').scale(0.7).move_to(4.5*RIGHT+1*UP)
        self.play(Write(uni))
        self.slide_break()

        diss_stp2 = SVGMobject('visco_figs/diss_toprow_stp2.svg').scale(scale).shift(shift)
        self.play(Write(diss_stp2))
        self.slide_break()

        diss_stp3 = SVGMobject('visco_figs/diss_toprow_stp3.svg').scale(scale).shift(shift)
        self.play(Write(diss_stp3))
        self.slide_break()

        diss_stp4 = SVGMobject('visco_figs/diss_toprow_stp4.svg').scale(scale).shift(shift)
        self.play(Write(diss_stp4))
        self.slide_break()

        diss_stp5 = SVGMobject('visco_figs/diss_toprow_stp5.svg').scale(scale).shift(shift)
        self.play(Write(diss_stp5))
        self.slide_break()

        diss_stp6 = SVGMobject('visco_figs/diss_toprow_stp6.svg').scale(scale).shift(shift)
        self.play(Write(diss_stp6), Write(equ), Write(labelset2))
        self.slide_break()

        diss_stp7 = SVGMobject('visco_figs/diss_toprow_stp7.svg').scale(scale).shift(shift)
        self.play(Write(diss_stp7), Write(ps), Write(labelset3))
        self.slide_break()

        self.play(FadeOut(diss_stp1, diss_stp2, diss_stp3, diss_stp4, diss_stp5, diss_stp6, 
                          diss_stp7, diss_label, labelset1, labelset2, labelset3, uni, equ, ps))
        

        scale = 2
        shift = 0.5*DOWN
        diss_stp1 = SVGMobject('visco_figs/diss_botrow_stp1.svg').scale(scale).shift(shift)
        diss_label = Tex('Dissipation').rotate(90*DEGREES).scale(0.7).move_to(5.5*LEFT+0.5*DOWN)
        self.play(Write(diss_stp1), Write(diss_label))
        self.slide_break()

        diss_stp2 = SVGMobject('visco_figs/diss_botrow_stp2.svg').scale(scale).shift(shift)
        self.play(Write(diss_stp2))
        self.slide_break()

        diss_stp3 = SVGMobject('visco_figs/diss_botrow_stp3.svg').scale(scale).shift(shift)
        self.play(Write(diss_stp3))
        self.slide_break()

        self.play(FadeOut(diss_stp1, diss_stp2, diss_stp3, diss_label))
        self.slide_break()

        brainfig = ImageMobject('visco_figs/fig_brain.png').scale(0.7)
        self.play(FadeIn(brainfig))
        self.slide_break()

        self.play(FadeOut(brainfig))


class damage(SlideScene):
    def construct(self):
        heading = toc[1].copy().move_to(ORIGIN).scale(1.25).to_corner(UP)
        self.add(heading)

        self.play(heading[3].animate.scale(1.2).set_color(YELLOW))
        self.play(heading[3].animate.scale(1/1.2).set_color(WHITE))

        dmg_title = Tex("\\begin{tabular}{c}Data-driven continuum damage mechanics with built-in physics\\textsuperscript{5}\\end{tabular}").scale(0.7).move_to(2.5*UP+1*LEFT)
        dmg_cite = Tex("\\begin{tabular}{c}\\textsuperscript{5}V. Tac, E. Kuhl, A. Buganza Tepole, Submitted, 2024.\\end{tabular}").scale(0.5).move_to(3.5*DOWN)

        self.play(Write(dmg_title))
        self.play(FadeIn(dmg_cite))

        Psi = MathTex("\Psi(\mathbf{C}, \mathbf{d}) = ", "f_1", "(", "d_1", ") \Psi_{1}^o(I_1) + ", "f_2", 
                      "(", "d_2", ") \Psi_{2}^o(I_2) + ", "f_3", "(", "d_3", ")\Psi_{3}^o(J) + \cdots").scale(0.75)
        self.play(Write(Psi))
        self.slide_break()

        self.play(*[Circumscribe(Psi[i]) for i in [3, 7, 11]])
        self.slide_break()

        self.play(*[Circumscribe(Psi[i]) for i in [1, 5, 9]])
        self.slide_break()

        self.play(Psi.animate.shift(1.5*UP))

        G_eqn = MathTex("\dot{d}_i &= \dot{\\tau}_i \\frac{\mathrm{d}G_i(\\tau_i)}{\mathrm{d}\\tau_i}").scale(0.75)
        self.play(Write(G_eqn))
        self.slide_break()

        G_mono = Tex("G: Monotonic, non-negative").scale(0.75).shift(DOWN + 2.5*LEFT)
        self.play(Write(G_mono))
        self.slide_break()

        node = Tex("$\\rightarrow$ Neural ODE").scale(0.75).next_to(G_mono, RIGHT)
        node[0][1:].set_color(YELLOW)
        self.play(Write(node))
        self.slide_break()

        self.play(FadeOut(Psi, G_eqn, G_mono, node))

        subq_fig = ImageMobject('dmg_figs/fig_subq.png').scale(1.2).shift(0.5*DOWN)
        self.play(FadeIn(subq_fig))
        self.slide_break()

        self.play(FadeOut(heading,  dmg_title, dmg_cite, subq_fig))



class diff_p1_context(SlideScene):
    def construct(self):

        self.add(toc, toctitle)

        heading = toc[2].copy().move_to(ORIGIN).scale(1.25).to_corner(UP)
        self.play(FadeOut(toc, toctitle), ReplacementTransform(toc[2],heading))
        self.slide_break()
        
        
        nodemodels = Tex("Previously: Hyperelasticity with NODEs").next_to(heading, DOWN, aligned_edge=LEFT)
        self.play(Create(nodemodels))

        physics = Group(
            Tex("\checkmark Polyconvex"),
            Tex("\checkmark Objective"),
            Tex("\checkmark Thermodynamically consistent"),
            Tex("\checkmark $\cdots$"),
            Tex("\checkmark Very flexible")
        ).scale(0.8).arrange(DOWN,aligned_edge=LEFT,buff=0.3).next_to(nodemodels, DOWN, aligned_edge=LEFT)
        for ph in physics:
            ph[0][0].set_color(GREEN)
            self.play(FadeIn(ph), run_time = 0.3)

        self.slide_break()
        self.remove(*physics)

        node1        = Rectangle(height=1.8, width=1.8).set_fill(orange1, opacity=1)
        nodelabel   = Tex("NODE").scale(0.8).set_color(BLACK)
        nodelabel_2   = Tex("NODE 1").scale(0.8).set_color(BLACK).shift(1.5*LEFT)
        theta1      = Tex("$\\boldsymbol{\\theta}$").set_color(BLACK).scale(0.8).shift(0.5*RIGHT+0.6*DOWN)
        theta1_2      = Tex("$\\boldsymbol{\\theta^{(1)}}$").set_color(BLACK).scale(0.8).shift(0.5*RIGHT+0.6*DOWN).shift(1.5*LEFT)
        deformation = Tex("Deformation $\\rightarrow$").scale(0.8).next_to(node1, LEFT)
        stress      = Tex("$\\rightarrow$ Stress").scale(0.8).next_to(node1, RIGHT)

        lmb = Tex("$\\boldsymbol{\\lambda} \\rightarrow$").scale(0.8).next_to(node1, LEFT)
        sgm = Tex("$\\rightarrow \\boldsymbol{\\sigma}$").scale(0.8).next_to(node1, RIGHT)

        self.play(Create(node1), Create(nodelabel), Create(theta1))
        self.play(Create(deformation), run_time=0.5)
        self.play(Create(stress), run_time=0.5)
        self.play(ReplacementTransform(deformation, lmb), ReplacementTransform(stress, sgm))
        self.slide_break()

        mouse1 = SVGMobject("diffusion_figs/mouse1.svg").scale(0.5).shift(3.5*LEFT+0.3*DOWN)
        mouse1label = Tex("Material 1").scale(0.8).next_to(mouse1, UP)
        self.play(FadeIn(mouse1, mouse1label))
        self.play(*[obj.animate.shift(1.5*LEFT) for obj in [lmb, sgm, node1, nodelabel, mouse1, mouse1label]], 
                  ReplacementTransform(nodelabel, nodelabel_2),
                  ReplacementTransform(theta1, theta1_2))



        with open('diffusion_params/anim_1_indiv.npy', 'rb') as f:
            lmb_gt, sgm_gt, sgm_pr = pickle.load(f)
        ax1 = Axes([1,1.3], [0,0.08], x_length=5, y_length=4).scale(0.4).shift(2*RIGHT)
        ax1_labels = ax1.get_axis_labels(
            Tex("$\\lambda_x$").scale(0.8), Tex("$\\sigma_{xx}$").scale(0.8)
        )
        data_label = Group(Tex("$\cdot$").scale(2), Tex("Data").scale(0.8)).arrange(RIGHT, buff=0.3).next_to(ax1, RIGHT).shift(1*UP+0.6*RIGHT)
        pred_label = Group(Tex("-").set_color(orange1).scale(2), Tex("Pred.").scale(0.8)).arrange(RIGHT, buff=0.3).next_to(data_label, DOWN, aligned_edge=LEFT)
        gt_graph1 = ax1.plot_line_graph(lmb_gt[0], 
                                       sgm_gt[0], 
                                       stroke_width=0, 
                                       vertex_dot_radius=0.03, 
                                       vertex_dot_style=dict(fill_color=WHITE))
        pr_graph1 = ax1.plot_line_graph(lmb_gt[0], 
                                       sgm_pr[0], 
                                       stroke_width=4, 
                                       add_vertex_dots=False, 
                                       line_color=orange1)
        self.play(Create(ax1), Create(ax1_labels))
        self.play(Create(gt_graph1))
        self.play(Create(data_label[0]), Create(data_label[1]))
        self.play(Create(pr_graph1))
        self.play(Create(pred_label[0]), Create(pred_label[1]))

        set1 = [mouse1, mouse1label, node1, nodelabel, nodelabel_2, lmb, sgm, ax1, ax1_labels, gt_graph1, pr_graph1, data_label, pred_label]

        """
        Create copies of this whole set
        """
        mouse2      = SVGMobject("diffusion_figs/mouse2.svg").scale(0.5).move_to(mouse1)
        mouse2label = Tex("Material 2").scale(0.8).move_to(mouse1label)
        node2       = Rectangle(height=1.8, width=1.8).set_fill(orange2, opacity=1).move_to(node1)
        node2label  = Tex("NODE 2").scale(0.8).set_color(BLACK).move_to(nodelabel_2)
        theta2      = Tex("$\\boldsymbol{\\theta^{(2)}}$").set_color(BLACK).scale(0.8).move_to(theta1_2)
        lmb2        = Tex("$\\boldsymbol{\\lambda} \\rightarrow$").scale(0.8).move_to(lmb)
        sgm2        = Tex("$\\rightarrow \\boldsymbol{\\sigma}$").scale(0.8).move_to(sgm)
        ax2         = Axes([1,1.3], [0,0.08], x_length=5, y_length=4).scale(0.4).move_to(ax1)
        ax2_labels  = ax2.get_axis_labels(Tex("$\\lambda_x$").scale(0.8), Tex("$\\sigma_{xx}$").scale(0.8))
        set2 = [mouse2, mouse2label, node2, node2label, lmb2, sgm2, ax2, ax2_labels, theta2]

        for obj in set2:
            obj.shift(1*RIGHT + 1*DOWN)

        gt_graph2 = ax2.plot_line_graph(lmb_gt[1], 
                                       sgm_gt[1], 
                                       stroke_width=0, 
                                       vertex_dot_radius=0.03, 
                                       vertex_dot_style=dict(fill_color=WHITE))
        pr_graph2 = ax2.plot_line_graph(lmb_gt[1], 
                                       sgm_pr[1], 
                                       stroke_width=4, 
                                       add_vertex_dots=False, 
                                       line_color=orange2)
        set2.append(gt_graph2)
        set2.append(pr_graph2)
        self.play(FadeIn(*set2))




        mouse3      = SVGMobject("diffusion_figs/mouse3.svg").scale(0.5).move_to(mouse2)
        mouse3label = Tex("Material 3").scale(0.8).move_to(mouse2label)
        node3       = Rectangle(height=1.8, width=1.8).set_fill(orange3, opacity=1).move_to(node2)
        node3label  = Tex("NODE 3").scale(0.8).set_color(BLACK).move_to(node2label)
        theta3      = Tex("$\\boldsymbol{\\theta^{(3)}}$").set_color(BLACK).scale(0.8).move_to(theta2)
        lmb3        = Tex("$\\boldsymbol{\\lambda} \\rightarrow$").scale(0.8).move_to(lmb2)
        sgm3        = Tex("$\\rightarrow \\boldsymbol{\\sigma}$").scale(0.8).move_to(sgm2)
        ax3         = Axes([1,1.3], [0,0.2], x_length=5, y_length=4).scale(0.4).move_to(ax2)
        ax3_labels  = ax3.get_axis_labels(Tex("$\\lambda_x$").scale(0.8), Tex("$\\sigma_{xx}$").scale(0.8))
        set3 = [mouse3, mouse3label, node3, node3label, lmb3, sgm3, ax3, ax3_labels, theta3]

        for obj in set3:
            obj.shift(1*RIGHT + 1*DOWN)

        gt_graph3 = ax3.plot_line_graph(lmb_gt[2], 
                                       sgm_gt[2], 
                                       stroke_width=0, 
                                       vertex_dot_radius=0.03, 
                                       vertex_dot_style=dict(fill_color=WHITE))
        pr_graph3 = ax3.plot_line_graph(lmb_gt[2], 
                                       sgm_pr[2], 
                                       stroke_width=4, 
                                       add_vertex_dots=False, 
                                       line_color=orange3)
        set3.append(gt_graph3)
        set3.append(pr_graph3)
        self.play(FadeIn(*set3))
        self.slide_break()


        self.play(FadeOut(*set1, *set2, *set3, nodemodels))


        mouse1 = SVGMobject('diffusion_figs/mouse1.svg').scale(0.5)
        mouse2 = SVGMobject('diffusion_figs/mouse2.svg').scale(0.5).shift(1*RIGHT+0.4*UP)
        mouse3 = SVGMobject('diffusion_figs/mouse3.svg').scale(0.5).shift(1*LEFT+0.8*DOWN)
        mouse4 = SVGMobject('diffusion_figs/mouse4.svg').scale(0.5).shift(0.2*RIGHT+1*UP)
        mouse5 = SVGMobject('diffusion_figs/mouse5.svg').scale(0.5).shift(0.4*RIGHT+0.4*DOWN)
        mouse6 = SVGMobject('diffusion_figs/mouse6.svg').scale(0.5).shift(1.2*LEFT)
        mice = [mouse1, mouse2, mouse3, mouse4, mouse5, mouse6]
        

        self.play(FadeIn(*mice))
        self.play(*[obj.animate.shift(1.5*LEFT) for obj in mice])
        ptheta = Tex("$p(\\boldsymbol{\\theta})$?").shift(1.5*RIGHT)
        deftheta = Tex("$\\boldsymbol{\\theta} \\equiv$ NODE parameters").scale(0.6).move_to(3.5*DOWN+5*LEFT)
        self.play(FadeIn(ptheta, deftheta))
        self.wait()

        self.slide_break()
        self.play(FadeOut(*mice, ptheta, deftheta))

class diff_p2_intro(SlideScene):
    def construct(self):
        heading = toc[2].copy().move_to(ORIGIN).scale(1.25).to_corner(UP)
        self.add(heading)

        # gener = Tex("Generative Hyperelasticity with Physics-Informed Diffusion\\textsuperscript{6}").scale(0.7).shift(2.5*UP+1.5*LEFT)
        # gener_cite = Tex("\\begin{tabular}{c}\\textsuperscript{6}V. Tac, M.K. Rausch, I. Bilionis, F. Sahli Costabal, A. Buganza Tepole, Submitted, 2023.\\end{tabular}").scale(0.5).move_to(3.5*DOWN)
        # self.play(Write(gener))
        # self.play(FadeIn(gener_cite))
        # self.slide_break()

        myshift = 0.5*LEFT
        diffbox = Rectangle(width=3, height=2).set_fill(color=BLUE, opacity=1).shift(myshift)
        difflabel = Tex(r"Diffusion \\ $p(\mathbf{x})$ \checkmark").set_color(BLACK).shift(myshift)
        diffinp = Tex("Data, $\mathbf{x} \\rightarrow$").scale(0.7).next_to(diffbox, LEFT)
        diffout = Tex("$\\rightarrow$ $\hat{\mathbf{x}} \sim p(\mathbf{x})$").scale(0.7).next_to(diffbox, RIGHT)

        self.play(FadeIn(diffbox, difflabel, diffinp, diffout, shift=UP))
        self.slide_break()



        """
        Examples of diffusion
        """
        chatgptlabel = SVGMobject('diffusion_figs/chatgpt_icon.svg').scale(0.4).shift(myshift)
        chatgptinp = Tex(r"Human language $\rightarrow$").scale(0.7).next_to(diffbox, LEFT)
        chatgptout = Tex(r"$\rightarrow$ Human-like language").scale(0.7).next_to(diffbox, RIGHT)
        self.play(ReplacementTransform(difflabel, chatgptlabel),
                  ReplacementTransform(diffinp, chatgptinp),
                  ReplacementTransform(diffout, chatgptout),)
        self.slide_break()


        dallelabel = Text("DALL·E 2").scale(0.8).shift(myshift).set_color(BLACK)
        dalleinp = Tex(r"Images $\rightarrow$").scale(0.7).next_to(diffbox, LEFT)
        dalleout = Tex(r"$\rightarrow$ Synthesized images").scale(0.7).next_to(diffbox, RIGHT)
        self.play(ReplacementTransform(chatgptlabel, dallelabel),
                  ReplacementTransform(chatgptinp, dalleinp),
                  ReplacementTransform(chatgptout, dalleout),)
        self.slide_break()

        self.play(FadeOut(dallelabel, dalleinp, dalleout, diffbox, dallelabel))
        self.slide_break()

 

        diffbox = Rectangle(width=7, height=5).shift(0.5*DOWN)
        difflabel = Tex("Diffusion").shift(1.5*UP)
        self.play(Create(diffbox), Create(difflabel))

        fwdsdebox = Rectangle(width=5, height=1).shift(0.5*UP).set_color(GREEN)
        fwdsdelabel = Tex("Forward SDE").shift(0.5*UP).set_color(GREEN)
        self.play(Create(fwdsdebox), Create(fwdsdelabel))

        rvssdebox = Rectangle(width=5, height=1).shift(1.5*DOWN).set_color(BLUE)
        rvssdelabel = Tex("Reverse SDE").shift(1.5*DOWN).set_color(BLUE)
        self.play(Create(rvssdebox), Create(rvssdelabel))
        self.slide_break()

        # inp_img = ImageMobject('figs/dog.jpeg').scale(0.2).next_to(fwdsdebox, LEFT).shift(1.5*LEFT)
        inp_img = SVGMobject('diffusion_figs/flower.svg').scale(0.8).next_to(fwdsdebox, LEFT).shift(1.5*LEFT)
        out_img = SVGMobject('diffusion_figs/noise1.svg').scale(0.6).next_to(fwdsdebox, RIGHT).shift(1.5*RIGHT)
        in_arrow = Arrow(start=LEFT, end=0.8*RIGHT).next_to(inp_img, RIGHT)
        out_arrow = Arrow(start=LEFT, end=0.8*RIGHT).next_to(out_img, LEFT)
        line1 = Line(start=0.5*DOWN, end=0.5*UP).move_to(2.5*LEFT+0.5*UP)
        self.play(FadeIn(inp_img))
        self.play(Create(in_arrow))
        self.play(ReplacementTransform(inp_img, line1), FadeOut(in_arrow))
        self.play(line1.animate.move_to(2.5*RIGHT+0.5*UP), run_time=3, rate_func=linear)
        self.play(ReplacementTransform(line1, out_img), FadeIn(out_arrow))
        self.slide_break()

        self.play(FadeOut(out_arrow, out_img))

        inp_img = SVGMobject('diffusion_figs/noise1.svg').scale(0.6).rotate(np.pi/2).next_to(rvssdebox, LEFT).shift(1.5*LEFT)
        out_img = SVGMobject('diffusion_figs/monalisa.svg').scale(0.8).next_to(rvssdebox, RIGHT).shift(1.5*RIGHT)
        in_arrow = Arrow(start=LEFT, end=0.8*RIGHT).next_to(inp_img, RIGHT)
        out_arrow = Arrow(start=LEFT, end=0.8*RIGHT).next_to(out_img, LEFT)
        line2 = Line(start=0.5*DOWN, end=0.5*UP).move_to(2.5*LEFT+1.5*DOWN)
        self.play(FadeIn(inp_img))
        self.play(Create(in_arrow))
        self.play(ReplacementTransform(inp_img, line2), FadeOut(in_arrow))
        self.play(line2.animate.move_to(2.5*RIGHT+1.5*DOWN), run_time=3, rate_func=linear)
        self.play(ReplacementTransform(line2, out_img), FadeIn(out_arrow))
        self.play(FadeOut(out_arrow, out_img))

        inp_img = SVGMobject('diffusion_figs/noise1.svg').scale(0.6).rotate(np.pi).next_to(rvssdebox, LEFT).shift(1.5*LEFT)
        out_img = SVGMobject('diffusion_figs/dali.svg').scale(0.8).next_to(rvssdebox, RIGHT).shift(1.5*RIGHT)
        in_arrow = Arrow(start=LEFT, end=0.8*RIGHT).next_to(inp_img, RIGHT)
        out_arrow = Arrow(start=LEFT, end=0.8*RIGHT).next_to(out_img, LEFT)
        line2 = Line(start=0.5*DOWN, end=0.5*UP).move_to(2.5*LEFT+1.5*DOWN)
        self.play(FadeIn(inp_img), run_time=0.5)
        self.play(Create(in_arrow), run_time=0.3)
        self.play(ReplacementTransform(inp_img, line2), FadeOut(in_arrow), run_time=0.5)
        self.play(line2.animate.move_to(2.5*RIGHT+1.5*DOWN), run_time=1.5, rate_func=linear)
        self.play(ReplacementTransform(line2, out_img), FadeIn(out_arrow), run_time=0.5)
        self.play(FadeOut(out_arrow, out_img), run_time=0.5)

        inp_img = SVGMobject('diffusion_figs/noise1.svg').scale(0.6).rotate(3*np.pi/2).next_to(rvssdebox, LEFT).shift(1.5*LEFT)
        out_img = SVGMobject('diffusion_figs/sunset.svg').scale(0.8).next_to(rvssdebox, RIGHT).shift(1.5*RIGHT)
        in_arrow = Arrow(start=LEFT, end=0.8*RIGHT).next_to(inp_img, RIGHT)
        out_arrow = Arrow(start=LEFT, end=0.8*RIGHT).next_to(out_img, LEFT)
        line2 = Line(start=0.5*DOWN, end=0.5*UP).move_to(2.5*LEFT+1.5*DOWN)
        self.play(FadeIn(inp_img), run_time=0.5)
        self.play(Create(in_arrow), run_time=0.3)
        self.play(ReplacementTransform(inp_img, line2), FadeOut(in_arrow), run_time=0.5)
        self.play(line2.animate.move_to(2.5*RIGHT+1.5*DOWN), run_time=1.5, rate_func=linear)
        self.play(ReplacementTransform(line2, out_img), FadeIn(out_arrow), run_time=0.5)
        # self.play(FadeOut(out_arrow, out_img), run_time=0.5)

        self.slide_break()
        self.play(FadeOut(diffbox, fwdsdebox, rvssdebox, difflabel, fwdsdelabel, rvssdelabel, out_arrow, out_img))


class diff_p3_diff4hyper(SlideScene):
    def construct(self):
        heading = toc[2].copy().move_to(ORIGIN).scale(1.25).to_corner(UP)
        self.add(heading)

        gener = Tex("Generative Hyperelasticity with Physics-Informed Diffusion\\textsuperscript{6}").scale(0.7).shift(2.7*UP+1.5*LEFT)
        gener_cite = Tex("\\begin{tabular}{c}\\textsuperscript{6}V. Tac, M.K. Rausch, I. Bilionis, F. Sahli Costabal, A. Buganza Tepole, Submitted, 2023.\\end{tabular}").scale(0.5).move_to(3.5*DOWN)
        self.play(Write(gener))
        self.play(FadeIn(gener_cite))
        self.slide_break()

        mouse1 = SVGMobject('diffusion_figs/mouse1.svg').scale(0.25)
        mouse2 = SVGMobject('diffusion_figs/mouse2.svg').scale(0.25).shift(0.5*RIGHT+0.2*UP)
        mouse3 = SVGMobject('diffusion_figs/mouse3.svg').scale(0.25).shift(0.5*LEFT+0.4*DOWN)
        mouse4 = SVGMobject('diffusion_figs/mouse4.svg').scale(0.25).shift(0.1*RIGHT+0.5*UP)
        mouse5 = SVGMobject('diffusion_figs/mouse5.svg').scale(0.25).shift(0.2*RIGHT+0.2*DOWN)
        mouse6 = SVGMobject('diffusion_figs/mouse6.svg').scale(0.25).shift(0.6*LEFT)
        mice = [mouse1, mouse2, mouse3, mouse4, mouse5, mouse6]
        for mouse in mice:
            mouse.shift(5*LEFT+1.5*UP)
        self.play(FadeIn(*mice))

        arrow1 = Tex("$\\rightarrow$").shift(4*LEFT+1.5*UP)
        self.play(FadeIn(arrow1))


        node1 = Rectangle(height=1.8, width=1.8).set_fill(orange1, opacity=1).scale(0.5)
        node2 = Rectangle(height=1.8, width=1.8).set_fill(orange2, opacity=1).scale(0.5)
        node3 = Rectangle(height=1.8, width=1.8).set_fill(orange6, opacity=1).scale(0.5)
        node4 = Rectangle(height=1.8, width=1.8).set_fill(orange4, opacity=1).scale(0.5)
        node5 = Rectangle(height=1.8, width=1.8).set_fill(orange5, opacity=1).scale(0.5)
        node6 = Rectangle(height=1.8, width=1.8).set_fill(orange3, opacity=1).scale(0.5)
        nodelabel = Tex("NODE").scale(0.5)
        thetalabel = Tex("$\\boldsymbol{\\theta^{(n)}}$").scale(0.25).shift(0.2*RIGHT+0.2*DOWN)
        nodes = [node1, node2, node3, node4, node5, node6, nodelabel, thetalabel]
        myshift = 0.1*RIGHT + 0.1*DOWN
        nodes[-1].shift(-myshift)
        nodes[-2].shift(-myshift)
        for i in range(len(nodes)):
            for j in range(i, len(nodes)):
                nodes[j].shift(myshift)
        for node in nodes:
            node.shift(3*LEFT+2.0*UP)
        self.play(FadeIn(*nodes))
        self.slide_break()


        arrow2 = Tex("$\\rightarrow$").shift(1.5*LEFT+1.5*UP)
        self.play(FadeIn(arrow2))

        ax1 = Axes([-1,31], [-3,3], x_length=6, y_length=4).scale(0.3).shift(1.5*UP)
        ax1_labels = ax1.get_axis_labels(
            "", Tex("Components of $\\boldsymbol{\\theta^{(i)}}$").scale(0.6)
        )
        with open('diffusion_params/mice_node_s_width_5.npy', 'rb') as f:
            Sample_params = pickle.load(f)
        w_diffusion = np.array([ravel_pytree(sample_params)[0] for sample_params in Sample_params])
        mu_x  = np.mean(w_diffusion,0)
        std_x = np.std (w_diffusion,0)
        w_diffusion_scaled = (w_diffusion-mu_x)/std_x

        self.play(Create(ax1), Create(ax1_labels))
        colors = [orange1, orange2, orange6, orange4, orange5, orange3]
        for i in range(6):
            s = w_diffusion_scaled[i]
            node = nodes[i]
            color = colors[i]
            graph = ax1.plot_line_graph(np.arange(len(s)), 
                                    s, 
                                    stroke_width=2,
                                    add_vertex_dots=False,
                                    line_color=color)
            self.play(ReplacementTransform(node.copy(), graph))

        arrow3 = Tex("$\\downarrow$").shift(0.5*UP)
        trainingdata = Tex("Training data for diffusion").scale(0.5).shift(0.5*UP+1.8*RIGHT)
        self.play(FadeIn(arrow3), Create(trainingdata))
        self.slide_break()

        rvsbox = Rectangle(width=5, height=1).set_color(BLUE).shift(0.5*DOWN)
        rvslabel = Tex("Reverse SDE").scale(0.8).set_color(BLUE).shift(0.5*DOWN)
        self.play(FadeIn(rvsbox, rvslabel))
        self.play(FadeOut(trainingdata))

        noiseax = Axes([-1,31], [-3,3], x_length=6, y_length=4).scale(0.3).shift(5*LEFT+0.5*DOWN)
        arrow4 = Tex("$\\rightarrow$").shift(3.5*LEFT+0.5*DOWN)
        noiseax_labels = noiseax.get_axis_labels(
            "", Tex("White Noise").scale(0.6)
        )
        outax = Axes([-1,31], [-3,3], x_length=6, y_length=4).scale(0.3).shift(5*RIGHT+0.5*DOWN)
        arrow5 = Tex("$\\rightarrow$").shift(3.5*RIGHT+0.5*DOWN)
        outax_labels = outax.get_axis_labels(
            "", Tex("Components of $\\boldsymbol{\\theta^{(i)}}$").scale(0.6)
        )


        self.play(FadeIn(noiseax, noiseax_labels, arrow4, arrow5, outax, outax_labels))

        params = []
        for i in range(5):
            speed = np.min([2.5,i+1])
            noise = np.random.normal(size=len(s))
            noise_graph = noiseax.plot_line_graph(np.arange(len(s)), 
                                                noise, 
                                                stroke_width=2,
                                                add_vertex_dots=False,
                                                line_color=WHITE)
            out_graph = outax.plot_line_graph(np.arange(len(s)), 
                                                w_diffusion_scaled[7+i], 
                                                stroke_width=2,
                                                add_vertex_dots=False,
                                                line_color=WHITE)
            line1 = Line(start=0.5*DOWN, end=0.5*UP).move_to(2.5*LEFT+0.5*DOWN)
            self.play(Create(noise_graph), run_time=1/speed)
            self.play(ReplacementTransform(noise_graph.copy(), line1), run_time=1/speed)
            self.play(line1.animate.move_to(2.5*RIGHT+0.5*DOWN), run_time=3/speed, rate_func=linear)
            self.play(ReplacementTransform(line1, out_graph), run_time=1/speed)
            params.append(out_graph)
        

        gray = "#808080"
        node1 = Rectangle(height=1.8, width=1.8).set_fill(gray, opacity=1).scale(0.5)
        node2 = Rectangle(height=1.8, width=1.8).set_fill(gray, opacity=1).scale(0.5)
        node3 = Rectangle(height=1.8, width=1.8).set_fill(gray, opacity=1).scale(0.5)
        node4 = Rectangle(height=1.8, width=1.8).set_fill(gray, opacity=1).scale(0.5)
        node5 = Rectangle(height=1.8, width=1.8).set_fill(gray, opacity=1).scale(0.5)
        nodelabel = Tex("NODE").scale(0.5)
        thetalabel = Tex("$\\boldsymbol{\\theta^{(n)}}$").scale(0.25).shift(0.2*RIGHT+0.2*DOWN)
        nodes = [node1, node2, node3, node4, node5, nodelabel, thetalabel]
        myshift = 0.1*RIGHT + 0.1*DOWN
        nodes[-1].shift(-myshift)
        nodes[-2].shift(-myshift)
        for i in range(len(nodes)):
            for j in range(i, len(nodes)):
                nodes[j].shift(myshift)
        for node in nodes:
            node.shift(2.2*DOWN)

        self.slide_break()

        for i in range(5):
            self.play(ReplacementTransform(params[i].copy(), nodes[i]), run_time=0.5)
        self.play(FadeIn(nodes[-1], nodes[-2]))
        
        stressax = Axes([1,1.25], [0,0.1], x_length=6, y_length=4).scale(0.3).next_to(nodes[-2], RIGHT).shift(RIGHT)
        stressax_labels = stressax.get_axis_labels(
            Tex("$\\boldsymbol{\\lambda}$").scale(0.6), Tex("$\\boldsymbol{\\sigma}$").scale(0.6)
        )
        arrow6 = Tex("$\\rightarrow$").next_to(stressax, LEFT)
        self.play(FadeIn(stressax, stressax_labels, arrow6))

        with open('diffusion_params/anim_1_stress.npy', 'rb') as f:
            lmbx_diff, sgmx_diff, lmbx_data, sgmx_data = pickle.load(f)

        for i in range(5):
            sgmx = sgmx_diff[-60][i] #The -60th ones just look better
            mask = sgmx<0.12
            graph = stressax.plot_line_graph(lmbx_diff[mask], 
                                            sgmx[mask], 
                                            stroke_width=2,
                                            line_color=WHITE,
                                            add_vertex_dots=False)
            self.play(Create(graph), run_time=0.5)

        pred_label = Group(Tex("-"), Tex("Predictions").scale(0.5)).arrange(RIGHT, buff=0.15).next_to(stressax, RIGHT).shift(0.2*UP+0.4*RIGHT)
        data_label = Group(Tex("-").set_color(orange3), Tex("Data").scale(0.5)).arrange(RIGHT, buff=0.15).next_to(pred_label, DOWN, aligned_edge=LEFT)
        self.play(Create(pred_label[0]), Create(pred_label[1]))
        self.slide_break()

        anims = []
        for lmbx, sgmx in zip(lmbx_data, sgmx_data):
            graph = stressax.plot_line_graph(lmbx, 
                                         sgmx, 
                                         stroke_width=1,
                                         add_vertex_dots=False,
                                         line_color=orange3)
            anims.append(Create(graph))
        self.play(*anims)
        self.play(Create(data_label[0]), Create(data_label[1]))

class diff_p4_fem(SlideScene):
    def construct(self):
        heading = toc[2].copy().move_to(ORIGIN).scale(1.25).to_corner(UP)

        gener = Tex("Generative Hyperelasticity with Physics-Informed Diffusion\\textsuperscript{6}").scale(0.7).shift(2.7*UP+1.5*LEFT)
        gener_cite = Tex("\\begin{tabular}{c}\\textsuperscript{6}V. Tac, M.K. Rausch, I. Bilionis, F. Sahli Costabal, A. Buganza Tepole, Submitted, 2023.\\end{tabular}").scale(0.5).move_to(3.5*DOWN)
        self.add(heading, gener, gener_cite)
        
        GaussianP = Tex("Gaussian Processes").set_color(YELLOW)
        self.play(Create(GaussianP))
        GP = Tex("GP").set_color(YELLOW)
        self.slide_break()

        self.play(ReplacementTransform(GaussianP, GP))

        GP_def1 = Tex("A ").next_to(GP,LEFT)
        GP_def2 = Tex(" is a").next_to(GP, RIGHT)
        GP_def3 = Tex("random process, from which we can sample functions.").next_to(GP,DOWN)
        self.play(FadeIn(GP_def1, GP_def2, GP_def3))
        self.slide_break()

        bullet1 = Tex("$\\cdot$").scale(2).next_to(GP_def3, DOWN, aligned_edge=LEFT)
        point1 = Tex(r"The fields have a specific spatial correlation given by a covariance \\ function $k(x,x')$.").scale(0.7).next_to(bullet1, RIGHT, aligned_edge=UP)
        self.play(FadeIn(bullet1, point1))

        bullet2 = Tex("$\\cdot$").scale(2).next_to(bullet1, DOWN, aligned_edge=LEFT).shift(0.5*DOWN)
        point2 = Tex(r"For a fixed $x=x^*$, the samples have a Normal distribution.").scale(0.7).next_to(bullet2, RIGHT, aligned_edge=UP)
        self.play(FadeIn(bullet2, point2))
        
        self.slide_break()

        self.play(FadeOut(GP, GP_def1, GP_def2, GP_def3, bullet1, bullet2, point1, point2))


        GP_ax = Axes([0,1], [-3,3], x_length=7, y_length=5).scale(0.6).shift(1.5*DOWN)
        GP_ax_labels = GP_ax.get_axis_labels(
            Tex("$x$"), Tex("$f(x)$")
        )
        self.play(Create(GP_ax), Create(GP_ax_labels))
        
        ### Sample a function from a GP
        def rbf_kernel(x1, x2, variance = 0.2):
            return np.exp(-1 * ((x1-x2) ** 2) / (2*variance))

        def gram_matrix(xs):
            return [[rbf_kernel(x1,x2) for x2 in xs] for x1 in xs]

        seed = 1
        rng = np.random.default_rng(seed)
        xs = np.arange(0, 1, 0.01)
        mean = [0 for x in xs]
        gram = gram_matrix(xs)
        ys = rng.multivariate_normal(mean, gram)
        GP_graph1 = GP_ax.plot_line_graph(xs, 
                                         ys, 
                                         stroke_width=3,
                                         add_vertex_dots=False)
        self.play(Create(GP_graph1))
        self.slide_break()


        GP_graphs = [GP_graph1]
        for i in range(9):
            ys = np.random.multivariate_normal(mean, gram)
            gp_graph = GP_ax.plot_line_graph(xs,
                                            ys,
                                            stroke_width=3,
                                            add_vertex_dots=False)
            GP_graphs.append(gp_graph)
        self.play(*[Create(obj) for obj in GP_graphs[1:]], run_time=2)
        self.play(*[obj.animate.shift(3*LEFT) for obj in [GP_ax, GP_ax_labels, *GP_graphs]])

        dashedline = DashedLine(1.5*UP, 1.5*DOWN, dash_length=0.2).shift(1.5*DOWN + 3*LEFT)
        self.play(Create(dashedline))

        dashedarrow = Arrow(1.5*DOWN, 1.5*DOWN + 2.5*RIGHT).shift(1.5*DOWN + 3*LEFT)
        self.play(Create(dashedarrow))


        noise_ax = Axes([0,1], [-3,3], x_length=7, y_length=5).scale(0.6).shift(1.5*DOWN + 3*RIGHT)
        noise_ax_labels = noise_ax.get_axis_labels(
            "", Tex("\"Noise\"")
        )
        self.play(Create(noise_ax), Create(noise_ax_labels))
        xs = np.linspace(0,1, 10)
        ys = np.random.normal(size=10)
        noise_graph = noise_ax.plot_line_graph(xs, ys, stroke_width=3)
        self.play(FadeIn(noise_graph))
        self.slide_break()


        self.play(FadeOut(GP_ax, GP_ax_labels, dashedline, dashedarrow, *GP_graphs))
        self.play(*[obj.animate.scale(0.6).shift(8*LEFT) for obj in [noise_ax, noise_ax_labels, noise_graph]])


        # passing = Tex("Passing these fields through a Reverse SDE\\textsuperscript{*} we get fields of material properties $\\theta(x)$.").scale(0.8).move_to(2*UP)
        rvssdebox = Rectangle(width=5, height=1).shift(1.5*DOWN).set_color(BLUE)
        rvssdelabel = Tex("Reverse SDE").shift(1.5*DOWN).set_color(BLUE)
        arrow1 = Tex("$\\rightarrow$").next_to(rvssdebox,LEFT)
        self.play(FadeIn(rvssdebox, rvssdelabel, arrow1))
        self.slide_break()
        
        arrow2 = Tex("$\\rightarrow$").next_to(rvssdebox,RIGHT)
        theta_ax = Axes([0,1], [-3,3], x_length=7, y_length=5).scale(0.6).scale(0.6).next_to(arrow2,RIGHT)
        theta_ax_labels = theta_ax.get_axis_labels(
            Tex("$x$"), Tex("$\\boldsymbol{\\theta}(x)$")
        )
        self.play(FadeIn(arrow2, theta_ax, theta_ax_labels))
        xs = np.arange(0, 1, 0.01)
        theta_graphs = []
        for i in range(10):
            ys = np.random.multivariate_normal(mean, gram)
            graph = theta_ax.plot_line_graph(xs,
                                            ys,
                                            stroke_width=3,
                                            add_vertex_dots=False)
            theta_graphs.append(graph)
        self.play(*[Create(graph) for graph in theta_graphs])
        self.slide_break()

        self.play(FadeOut(noise_ax, noise_ax_labels, noise_graph, *theta_graphs, theta_ax, theta_ax_labels, rvssdebox, rvssdelabel, arrow1, arrow2))


        # hetero = Tex("Using the same method we can generate 2D correlated fields and perform FE analysis of heterogeneous materials in Abaqus.").scale(0.8).move_to(2*UP)
        # self.play(Create(hetero))
        # self.wait(0.5)
        
        lenscale02 = Tex("Length scale 0.2L").scale(0.8).move_to(1*UP+5*LEFT)
        lenscale04 = Tex("Length scale 0.4L").scale(0.8).move_to(1*UP)
        lenscale06 = Tex("Length scale 0.6L").scale(0.8).move_to(1*UP+5*RIGHT)

        self.play(FadeIn(lenscale02, lenscale04, lenscale06))
        imgs1 = []
        for i in range(8):
            fname = "diffusion_figs/square_lenscale_10.0_init_1_param_" + str(i) + ".png"
            img = ImageMobject(fname).scale(0.7).next_to(lenscale02, DOWN).shift(i*(0.1*DOWN + 0.1*RIGHT))
            self.play(FadeIn(img), run_time=0.5)
            imgs1.append(img)
        
        imgs2 = []
        for i in range(8):
            fname = "diffusion_figs/square_lenscale_20.0_init_1_param_" + str(i) + ".png"
            img = ImageMobject(fname).scale(0.7).next_to(lenscale04, DOWN).shift(i*(0.1*DOWN + 0.1*RIGHT))
            self.play(FadeIn(img), run_time=0.5)
            imgs2.append(img)
        
        imgs3 = []
        for i in range(8):
            fname = "diffusion_figs/square_lenscale_30.0_init_1_param_" + str(i) + ".png"
            img = ImageMobject(fname).scale(0.7).next_to(lenscale06, DOWN).shift(i*(0.1*DOWN + 0.1*RIGHT))
            self.play(FadeIn(img), run_time=0.5)
            imgs3.append(img)

        self.slide_break()
        self.play(FadeOut(*imgs1, *imgs2, *imgs3, lenscale02, lenscale04, lenscale06, gener, gener_cite, heading))


class conclusion(SlideScene):
    def construct(self):
        conc = Tex('Conclusion').scale(0.7*1.25).to_corner(UP)
        self.play(Write(conc))
        self.slide_break()

        conc_1 = Tex('Use ', 'machine learning', ' methods').shift(1*UP+3*LEFT)
        conc_2 = Tex('to develop ', 'highly flexible', ' material models')
        conc_3 = Tex('that obey ', 'physics by design.').shift(1*DOWN+3*RIGHT)
        conc_1[1].set_color(YELLOW)
        conc_2[1].set_color(YELLOW)
        conc_3[1].set_color(YELLOW)

        self.play(FadeIn(conc_1))
        self.slide_break()
        self.play(FadeIn(conc_2))
        self.slide_break()
        self.play(FadeIn(conc_3))
        self.slide_break()

        self.play(FadeOut(conc, conc_1, conc_2, conc_3))

        twitter_logo = SVGMobject("Twitter-logo.svg").scale(0.25).shift(UP + LEFT)
        twitter_addr = Text("@tajtac").next_to(twitter_logo).shift(0.05*DOWN)
        web_logo = SVGMobject('web.svg').scale(0.25).set_color(GREEN).shift(LEFT)
        web_addr = Text("tajtac.com").next_to(web_logo).shift(0.05*DOWN)
        self.play(Write(twitter_logo))
        self.play(Write(twitter_addr))
        self.play(Write(web_logo))
        self.play(Write(web_addr))






