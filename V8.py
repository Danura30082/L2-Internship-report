# -*- coding: utf-8 -*-
"""
Created on Mon May 22 10:43:22 2023
@author: Arnaud Costermans
"""
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.patches as ptch
from scipy.integrate import solve_ivp
from wavelen2rgb import wavelen2rgb #importer de http://www.johnny-lin.com/py_refs/wavelen2rgb.html

#donnée issue de https://doi.org/10.1364/AO.20.000177 (Raymond C. Smith and Karen S. Baker (1981))

Liste_longeur_onde=[200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710, 720, 730, 740, 750, 760, 770, 780, 790, 800]
Liste_kappa=[3.07, 1.99, 1.31, 0.927, 0.72, 0.559, 0.457, 0.373, 0.288, 0.215, 0.141, 0.105, 0.0844, 0.0678, 0.0561, 0.0463, 0.0379, 0.03, 0.022, 0.0191, 0.0171, 0.0162, 0.0153, 0.0144, 0.0145, 0.0145, 0.0156, 0.0156, 0.0176, 0.0196, 0.0257, 0.0357, 0.0477, 0.0507, 0.0558, 0.0638, 0.0708, 0.0799, 0.108, 0.157, 0.244, 0.289, 0.309, 0.319, 0.329, 0.349, 0.4, 0.43, 0.45, 0.5, 0.65, 0.839, 1.169, 1.799, 2.38, 2.47, 2.55, 2.51, 2.36, 2.16, 2.07]
Liste_beta=[0.151, 0.119, 0.0995, 0.082, 0.0685, 0.0575, 0.0485, 0.0415, 0.0353, 0.0305, 0.0262, 0.0229, 0.02, 0.0175, 0.0153, 0.0134, 0.012, 0.0106, 0.0094, 0.0084, 0.0076, 0.0068, 0.0061, 0.0055, 0.0049, 0.0045, 0.0041, 0.0037, 0.0034, 0.0031, 0.0029, 0.0026, 0.0024, 0.0022, 0.0021, 0.0019, 0.0018, 0.0017, 0.0016, 0.0015, 0.0014, 0.0013, 0.0012, 0.0011, 0.001, 0.001, 0.0008, 0.0008, 0.0007, 0.0007, 0.0007, 0.0007, 0.0006, 0.0006, 0.0006, 0.0005, 0.0005, 0.0005, 0.0004, 0.0004, 0.0004]

def calcul_a_b_Isotrope(w, g):
    """    
    Calcule a et b, les coeficient de la matrice dans le cas isotropique

    Parameters
    ----------
    w : float
        valeur compris entre 0 et 1 qui represent l'albedo simple.
    g : float
        valeur compris entre -1 et 1 qui represent le parametre d'asymetrie.

    Returns
    -------
    a : float
        valeur compris entre 0 et 1 qui corespond a des coefficient de la matice.
    b : float
        valeur compris entre 0 et 1 qui corespond a des coefficient de la matice.

    """
    
    a=(1-w*((1+g)/2))
    b=(w*((1-g)/2))
    return a, b

def calcul_a_b_c_d_non_Isotrope(w,g):
    """
    Calcule a,b,c et d, les coeficient de la matrice dans le cas non-isotropique

    Parameters
    ----------
    w : float
        valeur compris entre 0 et 1 qui represent l'albedo simple.
    g : (float,float)
        Tuple contenant p_bas,bas et p_haut,haut compris entre 0 et 1.


    Returns
    -------
    a : float
        valeur compris entre 0 et 1 qui corespond a des coefficient de la matice.
    b : float
        valeur compris entre 0 et 1 qui corespond a des coefficient de la matice.
    c : float
        valeur compris entre 0 et 1 qui corespond a des coefficient de la matice.
    d : float
        valeur compris entre 0 et 1 qui corespond a des coefficient de la matice.

    """
    e=g[0]
    f=g[1]
    a=(w*e-1)
    b=(w*(1-f))
    c=-w*(1-e)
    d=(-w*f+1)
    return a,b,c,d

def modelflux(tau, Fbas_Fhaut,w,g):
    """
    Permet de calculer Fbas' et Fhaut'. Cette fonction est uniqument appeller par solve_ivp de la librairie scipy.integrate
    
    Parameters
    ----------
    tau : float
        l'épaiseur optique.
    Fbas_Fhaut : tuple [float,float]
        Vecteur composer de Fbas et de Fhaut.
    w : float
        valeur compris entre 0 et 1 qui represent l'albedo simple.
    g : float or (float,float)
        Soit valeur compris entre -1 et 1 qui represent le parametre d'asymetrie 
        ou un Tuple contenant p_bas,bas et p_haut,haut compris entre 0 et 1.

    Returns
    -------
    list : tuple [float,float]
        Vecteur composer de Fbas' et de Fhaut'.

    """
    Fbas, Fhaut= Fbas_Fhaut
    if type(g)==tuple:
        a,b,c,d=calcul_a_b_c_d_non_Isotrope(w, g)
        return [ a*Fbas+b*Fhaut, c*Fbas+d*Fhaut]
    else:
        a,b = calcul_a_b_Isotrope(w, g)
        return [ -a*Fbas+b*Fhaut, -b*Fbas+a*Fhaut]


def graph_des_phase(w, g, color="#1f77b4"):
    """
    Génère le diagrame de phase 

    Parameters
    ----------
    w : float
        valeur compris entre 0 et 1 qui represent l'albedo simple.
    g : float or (float,float)
        Soit valeur compris entre -1 et 1 qui represent le parametre d'asymetrie 
        ou un Tuple contenant p_bas,bas et p_haut,haut compris entre 0 et 1.
    color: string or (float,float,float)
        un argument couleur de mathplotlib ou une valeur RVB. The default is "black".
    Returns
    -------
    None.

    """
    #creation des parametres de streamplot
    Y, X = np.mgrid[0:1:200j, 0:1:200j] 
    if type(g)==tuple:
        a,b,c,d=calcul_a_b_c_d_non_Isotrope(w, g)
        U = a*X+b*Y 
        V = c*X+d*Y
    else:
        a,b = calcul_a_b_Isotrope(w, g)
        U = -a*X+b*Y 
        V = -b*X+a*Y
    
    #tracage du digramme des phases
    plt.streamplot(X, Y, U, V, density = 1,arrowsize=3,color=color)

def solution_particuliere (Fbas_init, Fhaut_init, w, g, color="black", resolution=1000, tau_min=0, tau_max=20):
    """
    Permet de tracer une solution particuliaire 

    Parameters
    ----------
    Fbas_init : float
        La valeur initial de Fbas. Compris entre 0 et 1.
    Fhaut_init : float
        La valeur initial de Fhaut. Compris entre 0 et 1.
    w : float
        valeur compris entre 0 et 1 qui represent l'albedo simple.
    g : float or (float,float)
        Soit valeur compris entre -1 et 1 qui represent le parametre d'asymetrie 
        ou un Tuple contenant p_bas,bas et p_haut,haut compris entre 0 et 1.
    color: string or (float,float,float)
        un argument couleur de mathplotlib ou une valeur RVB. The default is "black".
    resolution : int, optional
        Le nombre de point qui seront tracer pour la solution particuliaire. The default is 1000.
    tau_min : int, optional
        La valuer initial pour la derivation par tau, l'epaiseur optique. The default is 0.
    tau_max : int, optional
        La valuer final pour la derivation par tau, l'epaiseur optique. The default is 20.

    Returns
    -------
    None.

    """
    
    
    #calcule de la solution particuliaire
    solution=solve_ivp(modelflux, [tau_min,tau_max], [Fbas_init,Fhaut_init], t_eval=np.linspace(tau_min,tau_max,resolution), args=(w,g))

    #tracage de la solution particuliere
    plt.plot(solution.y[0], solution.y[1], '-',color=color, lw=3)
    

def point_particulier(Fbas_init, Fhaut_init, w, g, tau=0, resolution=1000, tau_min=0, tau_max=20):
    """
    
    Permet de tracer un point sur une solution particuliaire
    Parameters
    ----------
    Fbas_init : float
        La valeur initial de Fbas. Compris entre 0 et 1.
    Fhaut_init : float
        La valeur initial de Fhaut. Compris entre 0 et 1.
    w : float
        valeur compris entre 0 et 1 qui represent l'albedo simple.
    g : float or (float,float)
        Soit valeur compris entre -1 et 1 qui represent le parametre d'asymetrie 
        ou un Tuple contenant p_bas,bas et p_haut,haut compris entre 0 et 1.
    tau : int, optional
        La valeur de tau, l'epaisseur optique, a laquelle on s'arrete. The default is 0.
    resolution : int, optional
        Le nombre de point qui seront tracer pour la solution particuliaire. The default is 1000.
    tau_min : int, optional
        La valuer initial pour la derivation par tau, l'epaiseur optique. The default is 0.
    tau_max : int, optional
        La valuer final pour la derivation par tau, l'epaiseur optique. The default is 20.

    Returns
    -------
    solution: OdeResult object of scipy.integrate._ivp.ivp
        Une solution particuliere continue 
        

    """
    
    #calcule de la solution particuliaire
    solution=solve_ivp(modelflux, [tau_min,tau_max], [Fbas_init,Fhaut_init], dense_output=True, t_eval=np.linspace(tau_min,tau_max,resolution), args=(w,g))
    plt.plot(solution.sol.__call__(tau)[0], solution.sol.__call__(tau)[1], 'k.', ms=20)
    return solution
def vecteur_propre(w,g,typ='r-'):
    """
    Permet de dessiner les vecteur propre sur le graphe

    Parameters
    ----------
    w : float
        valeur compris entre 0 et 1 qui represent l'albedo simple.
    g : float or (float,float)
        Soit valeur compris entre -1 et 1 qui represent le parametre d'asymetrie 
        ou un Tuple contenant p_bas,bas et p_haut,haut compris entre 0 et 1.
    typ: string 
        Un format String conforme a la documentation Mathplotlib. The default is 'r-'.

    Returns
    -------
    None.

    """
    if type(g)==tuple:
        a,b,c,d=calcul_a_b_c_d_non_Isotrope(w, g)
        e=g[0]
        f=g[1]
    else:
        a,b = calcul_a_b_Isotrope(w, g)
        e=(g+1)/2
        f=e
    if w!=0:# empeche la division pas 0
        x_v1 = [0, -(2 - e*w - f*w + np.sqrt(4 - 4*e*w - 4*f*w - 4*(w**2) + 4*e*(w**2) + (e**2)*(w**2) + 4*f*(w**2) - 2*e*f*(w**2) + (f**2)*(w**2)))/(2*(-1 + e)*w)]
        y_v1 = [0, 1]
        x_v2 = [0,-(2 - e*w - f*w - np.sqrt(4 - 4*e*w - 4*f*w - 4*(w**2) + 4*e*(w**2) + (e**2)*(w**2) + 4*f*(w**2) - 2*e*f*(w**2) + (f**2)*(w**2)))/(2*(-1 + e)*w)]
        y_v2 = [0, 1]
    elif type(g)!=tuple:
        alpha = np.sqrt(a**2 - b**2)
        x_v1 = [0, b*3]
        y_v1 = [0, (a+alpha)*3]
        y_v2=x_v1
        x_v2=y_v1
    else:
        print("ERROR: w=0 et cas non isotrope pas supporté")
        return
    plt.plot(x_v1,y_v1,typ,lw=4)
    plt.plot(x_v2,y_v2,typ,lw=4)
    

def mise_en_forme():
    """
    Permet de mettre en forme le graphique

    Returns
    -------
    None.

    """
    
    #parametrage des axes
    label=["0","0.2","0.4","0.6","0.8","1"]
    plt.tick_params(labelsize=24)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.ylabel(r'$F_{\!\uparrow}$', fontsize=30,  rotation='horizontal',labelpad=20.0 ,y=0.45)
    plt.xlabel(r'$F_{\!\downarrow}$', fontsize=30,labelpad=-15.0)
    plt.xticks(np.linspace(0,1,6),label)
    plt.yticks(np.linspace(0,1,6),label)

def arc_en_ceil(Fbas_init, Fhaut_init,g,longeur_onde,tau_max=20):
    """
    Permet de tracer une solution particulier pour un w corespondant a une certain longeru d'onde.

    Parameters
    ----------
    Fbas_init : float
        La valeur initial de Fbas. Compris entre 0 et 1.
    Fhaut_init : float
        La valeur initial de Fhaut. Compris entre 0 et 1.
    g : float or (float,float)
        Soit valeur compris entre -1 et 1 qui represent le parametre d'asymetrie 
        ou un Tuple contenant p_bas,bas et p_haut,haut compris entre 0 et 1.
    longeur_onde : int
        La longeur d'onde a tracer avec son w respectif.

    Returns
    -------
    None.

    """
    
    if not(longeur_onde<380 or 780<longeur_onde):
        lamda=Liste_longeur_onde.index(longeur_onde)
        w=Liste_beta[lamda]/(Liste_beta[lamda]+Liste_kappa[lamda])
        solution_particuliere(Fbas_init, Fhaut_init, w, g, color=tuple([float(x/100) for x in wavelen2rgb(Liste_longeur_onde[lamda])]),tau_max=tau_max)
         
def point_final (Fbas_init, Fhaut_init, w, g, seuil_max=1, seuil_min=0, resolution=1000, tau_min=0, tau_max=20):
    """
    Cette fonction permet de trouver le point sur une solution particulier qui dépasse un certain seuil de Fhaut

    Parameters
    ----------
    Fbas_init : float
        La valeur initial de Fbas. Compris entre 0 et 1.
    Fhaut_init : float
        La valeur initial de Fhaut. Compris entre 0 et 1.
    w : float
        valeur compris entre 0 et 1 qui represent l'albedo simple.
    g : float or (float,float)
        Soit valeur compris entre -1 et 1 qui represent le parametre d'asymetrie 
        ou un Tuple contenant p_bas,bas et p_haut,haut compris entre 0 et 1.
    seuil_max : float, optional
        Le seuil supérieur au-delà duquel on arrêt la solution. The default is 1.
    seuil_min : float, optional
        Le seuil inférieur en dessous duquel on arrêt la solution. The default is 0.
    resolution : int, optional
        Le nombre de point qui seront tracer pour la solution particuliaire. The default is 1000.
    tau_min : int, optional
        La valuer initial pour la derivation par tau, l'epaiseur optique. The default is 0.
    tau_max : int, optional
        La valuer final pour la derivation par tau, l'epaiseur optique. The default is 20.

    Returns
    -------
    tau : float
        La valeur de tau au seuil où on arrêt la solution.
    Fbas : float
        La valeur de Fbas au seuil où on arrêt la solution.
    Fhaut : float
        La valeur de Fhaut au seuil où on arrêt la solution.

    """
    
    
    #calcule de la solution particuliaire
    solution=solve_ivp(modelflux, [tau_min,tau_max], [Fbas_init,Fhaut_init], dense_output=True, t_eval=np.linspace(tau_min,tau_max,resolution), args=(w,g))
    for loop in range (len(solution.t)):
        if solution.y[1][loop]>seuil_max:
            if abs(seuil_max-solution.y[1][loop])<abs(seuil_max-solution.y[1][loop-1]):
                endpoint=loop
                break
            else:
                endpoint=loop-1
                break
        elif solution.y[1][loop]<seuil_min:
            if abs(seuil_min-solution.y[1][loop])<abs(seuil_min-solution.y[1][loop-1]):
                endpoint=loop
                break
            else:
                endpoint=loop-1
                break
        endpoint=loop
    Fbas=solution.y[0][endpoint]
    Fhaut=solution.y[1][endpoint]
    tau=solution.t[endpoint]
    return tau, Fbas , Fhaut

# %% w=0 Figure 4

w, g, Fbas_init, Fhaut_init=0, 0, 1, 0.2 #definition des parametre de cette figure
fig = plt.figure(figsize = (12, 7))  #creation du fond de la figure
mise_en_forme()   #mise en forme des axe et des legende  
graph_des_phase(w, g)    #Crée le portrait de phase 
solution_particuliere(Fbas_init, Fhaut_init, w, g)  #permet de tracer une solution particulier qui part de Fbas_init, Fhaut_init
vecteur_propre(w, g) #permet de tracer les vecteur propre de la matrice
tau_final, Fbas_final, Fhaut_final = point_final(Fbas_init, Fhaut_init, w, g) #on recupére les cordonné du point final
point_particulier(Fbas_init, Fhaut_init, w, g, tau_final) #on trace le point final

#on crée la legende du graphe
legende="Avec w=" + str(w) + " et g=" + str(g) +". Pour " + r'$\tau=$' + str(round(tau_final,2)) + ", " + r'$F_{\!\downarrow}=$' + str(round(Fbas_final,1)) + " et " + r'$F_{\!\uparrow}=$'+str(round(Fhaut_final,1))  
plt.text(.5, -0.25, legende, fontsize=24, color='black', ha='center')
plt.show()

# %% w=1 Figure 5

w, g, Fbas_init, Fhaut_init=1, 0, 1, 0.5  #definition des parametre de cette figure
fig = plt.figure(figsize = (12, 7)) #creation du fond de la figure
mise_en_forme()   #mise en forme des axe et des legende
graph_des_phase(w, g)    #Crée le portrait de phase 
solution_particuliere(Fbas_init, Fhaut_init, w, g) #permet de tracer une solution particulier qui part de Fbas_init, Fhaut_init
vecteur_propre(w, g) #permet de tracer les vecteur propre de la matrice
tau_final, Fbas_final, Fhaut_final = point_final(Fbas_init, Fhaut_init, w, g) #on recupére les cordonné du point final
point_particulier(Fbas_init, Fhaut_init, w, g, tau_final) #on trace le point final

#on crée la legende du graphe
legende="Avec w=" + str(w) + " et g=" + str(g) +". Pour " + r'$\tau=$' + str(round(tau_final,2)) + ", " + r'$F_{\!\downarrow}=$' + str(round(Fbas_final,1)) + " et " + r'$F_{\!\uparrow}=$'+str(round(Fhaut_final,1))  
plt.text(.5, -0.25, legende, fontsize=24, color='black', ha='center')
plt.show()

# %% w=0.7 Figure 6

w, g, Fbas_init, Fhaut_init=0.7, 0.2, 1, 0.5  #definition des parametre de cette figure
fig = plt.figure(figsize = (12, 7)) #creation du fond de la figure
mise_en_forme()   #mise en forme des axe et des legende
graph_des_phase(w, g)    #Crée le portrait de phase 
solution_particuliere(Fbas_init, Fhaut_init, w, g) #permet de tracer une solution particulier qui part de Fbas_init, Fhaut_init
vecteur_propre(w, g) #permet de tracer les vecteur propre de la matrice
tau, Fbas_final, Fhaut_final = point_final(Fbas_init, Fhaut_init, w, g, tau_max=1.25 ) #on recupére les cordonné du point qui correspond au schema
point_particulier(Fbas_init, Fhaut_init, w, g, tau) #on trace le point qui correspond au schema

#on crée la legende du graphe
legende="Avec w=" + str(w) + " et g=" + str(g) +". Pour " + r'$\tau=$' + str(round(tau_final,2)) + ", " + r'$F_{\!\downarrow}=$' + str(round(Fbas_final,2)) + " et " + r'$F_{\!\uparrow}=$'+str(round(Fhaut_final,2))  
plt.text(.5, -0.25, legende, fontsize=24, color='black', ha='center')
plt.show()

# %% non-iso Figure 7

w,g=0.7,(0.3,0.9)  #definition des parametre de cette figure
fig = plt.figure(figsize = (12, 7)) #creation du fond de la figure
mise_en_forme()   #mise en forme des axe et des legende
graph_des_phase(w, g)    #Crée le portrait de phase 
solution_particuliere(1, 0.6, w, g) #permet de tracer une solution particulier qui part de Fbas_init, Fhaut_init
vecteur_propre(w, g) #permet de tracer les vecteur propre de la matrice

#on crée la legende du graphe
legende="Avec w=" + str(w) + " et pour " + r'$p_{\!\downarrow\!\downarrow}=$' + str(g[0]) +", "+r'$p_{\!\uparrow\!\uparrow}=$'+str(g[1]) 
plt.text(.5, -0.25, legende, fontsize=24, color='black', ha='center')
plt.show()

# %% conservation du volume Figure 8

w, g, Fbas_init, Fhaut_init=0.7, 0.2, 0.75, 0.4  #definition des parametre de cette figure
fig = plt.figure(figsize = (12, 7)) #creation du fond de la figure
mise_en_forme()   #mise en forme des axe et des legende
graph_des_phase(w, g)    #Crée le portrait de phase 
vecteur_propre(w, g) #permet de tracer les vecteur propre de la matrice

#on trace les 8 point a tau=0
for loop in range (3):
    for loop1 in range (3):
        if loop!=1 or loop1!=1: #on empeche le tracage du point central
            point_particulier(Fbas_init+loop1/10, Fhaut_init+loop/10, w, g)
            
#on trace les 8 point a tau=1.25  
for loop in range (3):
    for loop1 in range (3):
        if loop!=1 or loop1!=1: #on empeche le tracage du point central
            point_particulier(Fbas_init+loop1/10, Fhaut_init+loop/10, w, g,1.25)
            
#on crée la legende du graphe
legende="Avec w=" + str(w) + " et g=" + str(g) 
plt.text(.5, -0.25, legende, fontsize=24, color='black', ha='center')
plt.show()

# %% non-conservation du volume Figure 9

w, g, Fbas_init, Fhaut_init=0.7, (0.3,0.9), 0.75, 0.45  #definition des parametre de cette figure
fig = plt.figure(figsize = (12, 7)) #creation du fond de la figure
mise_en_forme()   #mise en forme des axe et des legende
graph_des_phase(w, g)    #Crée le portrait de phase 
vecteur_propre(w, g) #permet de tracer les vecteur propre de la matrice

#on trace les 8 point a tau=0
for loop in range (3):
    for loop1 in range (3):
        if loop!=1 or loop1!=1:  #on empeche le tracage du point central
            point_particulier(Fbas_init+loop1/10, Fhaut_init+loop/10, w, g)


#on trace les 8 point a tau=2  
for loop in range (3):
    for loop1 in range (3):
        if loop!=1 or loop1!=1:  #on empeche le tracage du point central
            point_particulier(Fbas_init+loop1/10, Fhaut_init+loop/10, w, g,2)
            
#on crée la legende du graphe
legende="Avec w=" + str(w) + " et pour " + r'$p_{\!\downarrow\!\downarrow}=$' + str(g[0]) +", "+r'$p_{\!\uparrow\!\uparrow}=$'+str(g[1]) 
plt.text(.5, -0.25, legende, fontsize=24, color='black', ha='center')
plt.show()


# %% longeur d'onde aire figure 11
w=0   #definition des parametre de cette figure
g=0
fig = plt.figure(figsize = (12, 7)) #creation du fond de la figure
mise_en_forme()   #mise en forme des axe et des legende
kappa_450=0.02
kappa_675=kappa_450/5
beta_450=0.1
beta_675=beta_450/5
w_450=beta_450/(beta_450+kappa_450)
w_675=beta_675/(beta_675+kappa_675)
vecteur_propre(w_450, g,typ="black") #permet de tracer les vecteur propre de la matrice
graph_des_phase(w_450, g, "#1f77b4")
tau=0.25
solution_particuliere(1, 0.30, w_450, g, color=tuple([float(x/100) for x in wavelen2rgb(Liste_longeur_onde[Liste_longeur_onde.index(450)])]),tau_max=5*tau)
solution_particuliere(1, 0.30, w_675, g, color=tuple([float(x/100) for x in wavelen2rgb(Liste_longeur_onde[Liste_longeur_onde.index(670)])]),tau_max=tau)
tau_final, Fbas_final, Fhaut_final = point_final(1, 0.3, w_450, g,tau_max=5*tau)
print("tau=",tau_final,"z=",tau_final/(beta_450+kappa_450))
tau_final, Fbas_final, Fhaut_final = point_final(1, 0.3, w_675, g,tau_max=tau)
print("tau=",tau_final,"z=",tau_final/(beta_675+kappa_675))
solution_particuliere(1, 0.57, w_450, g, color=tuple([float(x/100) for x in wavelen2rgb(Liste_longeur_onde[Liste_longeur_onde.index(450)])]),tau_max=5*tau)
solution_particuliere(1, 0.57, w_675, g, color=tuple([float(x/100) for x in wavelen2rgb(Liste_longeur_onde[Liste_longeur_onde.index(670)])]),tau_max=tau)
solution_particuliere(0, 0.50, w_450, g, color=tuple([float(x/100) for x in wavelen2rgb(Liste_longeur_onde[Liste_longeur_onde.index(450)])]),tau_max=5*tau)
solution_particuliere(0, 0.50, w_675, g, color=tuple([float(x/100) for x in wavelen2rgb(Liste_longeur_onde[Liste_longeur_onde.index(670)])]),tau_max=tau)
print(w_450,w_675)
 
# %% nuage Figure 14
"""graphe nuage"""
w, g ,tau,Fbas_init, Fhaut_init= 0.8,-1, 0.75,1,0.6  #definition des parametre de cette figure
fig = plt.figure(figsize = (12, 7)) #creation du fond de la figure
graph_des_phase(w, g, "#1f77b4")  
mise_en_forme()   #mise en forme des axe et des legende

#on hachure sur la partie du graphe interdite
plt.fill_between([0,1], [0,1],[1,1],facecolor="w", hatch="/", edgecolor="k", linewidth=0.0,zorder=100)
plt.plot([0,1],[0,1],'k--')
plt.plot([0,0,1],[0,1,1],'k-',zorder=101)


vecteur_propre(w, g,'b') #permet de tracer les vecteur propre de la matrice en noir
solution_particuliere(Fbas_init, Fhaut_init, w, g,tau_max=tau) #on trace notre solution particulier avec 
tau,albedoX,albedoY=point_final(Fbas_init, Fhaut_init, w, g,tau_max=tau) #on recupere les cordoner du point final
plt.plot([0,albedoX*3],[0,albedoY*3],color=(0,1,0),lw=4) #on trace la droite qui passe par l'originer et le point final
plt.plot([albedoX,albedoX],[0,albedoY],'k--',lw=3) #on trace la ligne pointilleer qui relie le point final a l'absice

#on change la mise en forme des axe
plt.ylabel(r'$\frac{F_{\!\uparrow}}{F_{\!\downarrow\!0}}$', fontsize=40,  rotation='horizontal',labelpad=20.0 ,y=0.6)
plt.xlabel(r'$\frac{F_{\!\downarrow}}{F_{\!\downarrow\!0}}$', fontsize=40,labelpad=-15.0)
ax=plt.gca()
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")

#plt.text()

# montrer la figure
plt.show()

# %% Beta Kappa Figure 10

fig = plt.figure(figsize = (12, 7)) #creation du fond de la figure

#on met en forme les axe
plt.tick_params(labelsize=24)
plt.ylabel(r'$\kappa\,(m^{-1})$', fontsize=30,  labelpad=20.0)
plt.xlabel(r'$\lambda\,(nm)$', fontsize=30)

ax1=plt.gca()
plt.tick_params(labelsize=24)
ax2 = ax1.twinx() #on crée le deuxime axe des ordonnée
plt.ylabel(r'$\beta (m^{-1})$', fontsize=30,  labelpad=20.0)
plt.tick_params(labelsize=24)

#on crée les courbe de beta et de kappa
kappa,=ax1.plot(Liste_longeur_onde,Liste_kappa,lw=4)
beta,=ax2.plot(Liste_longeur_onde,Liste_beta,'r',lw=4)

#on cree la legende
ax1.legend([kappa,beta],[r'$\kappa$',r'$\beta $'],fontsize="20",loc=9)
plt.show()

# %% arc en ciel eau z fixé Figure 10

w, g, z= 0,-1, 50  #definition des parametre de cette figure
fig = plt.figure(figsize = (12, 7)) #creation du fond de la figure
mise_en_forme()   #mise en forme des axe et des legende
#in initialiser les liste qui permetrond de tracer les tau finaux
tau_final=[]
Longeur_onde_trace=[]
# on iter a traver la liste en tracant que les longeur d'onde qui on une couleur corsspondant
for loop in range (len(Liste_longeur_onde)):
    if not(Liste_longeur_onde[loop]<400 or 780<Liste_longeur_onde[loop]):
        lamda=Liste_longeur_onde.index(Liste_longeur_onde[loop])
        tau=z*(Liste_beta[lamda]+Liste_kappa[lamda]) #on calcule le tau associer a la longeur z pour cette longeur d'onde
        arc_en_ceil(1, 0.2, g, Liste_longeur_onde[loop],tau)
        tau_final.append(tau)
        Longeur_onde_trace.append(Liste_longeur_onde[loop])

#on trace le vecteur propre acosier a la plus petit longeur d'onde
vecteur_propre(Liste_beta[Liste_longeur_onde.index(400)]/(Liste_beta[Liste_longeur_onde.index(400)] + Liste_kappa[Liste_longeur_onde.index(400)]), g,'k-') 

# creation de la figure des tau final dans l'eau
fig = plt.figure(figsize = (12, 7)) #creation du fond de la figure     
plt.plot(Longeur_onde_trace,tau_final)
plt.ylabel(r'$\tau_{final}}$', fontsize=30)
plt.xlabel(r'$\lambda (nm)$', fontsize=30)
plt.tick_params(labelsize=24)