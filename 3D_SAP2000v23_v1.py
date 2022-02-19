# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 01:01:26 2021

@author: hakan
"""

##################################################################
## 3D frame example to show how to render opensees model and 
## plot mode shapes
##
## By - Hakan Keskin, PhD Student, Istanbul Technical University.
## Updated - 23/10/2021
##################################################################

import openseespy.postprocessing.Get_Rendering as opsplt
import openseespy.opensees as ops
import streamlit as st
from bokeh.plotting import figure
from pandas import read_excel
import pandas as pd
import numpy as np
import pydeck as pdk
from PIL import Image
from math import asin, sqrt
import openseespy.postprocessing.ops_vis as opsv
import matplotlib.pyplot as plt
import openseespy.opensees as ops
import openseespy.postprocessing.ops_vis as opsv

st.title("Basic - 3D RC System Modelling ")

"""
This application was developed to find out capacity and DCR of IPE-HE
sections.

To use the app you should choose at the following items below:
    
    1- Results type
    2- Section
    3- Steel material
    4- Design type
    5- Length
    6- Unbraced length factors
    
Also if you want to calculate DCR of section, you can use design forces 
end of the page. Be sure the choice correct "Result" input. 
            
To find the more information about these parameters please check the "ANS/AISC 360-16
Specifacation for Structural Steel Buildings". Also you can download the AISC at the following
link: https://www.aisc.org/globalassets/aisc/publications/standards/a360-16w-rev-june-2019.pdf
        
"""
section_rebar = "Frame Props 02 - Concrete Col" # change it to the name of your excel file
section_properties = 'Frame Props 01 - General'
restrained = 'Joint Restraint Assignments'
frame_section = 'Frame Section Assignments'
assembled_joint_masses = 'Assembled Joint Masses' # change it to your sheet name
connectivity_frame = "Connectivity - Frame"
joint_coordinates = "Joint Coordinates"

import streamlit as st
import os

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

filename = file_selector()
st.write('You selected `%s`' % filename)

main_file_name = filename    
# main_file_name = '3D_Example_Deneme.xlsx' # change it to the name of your excel file

# df_main = read_excel(main_file_name, sheet_name = my_sheet_main, engine='openpyxl')
df_joint_mass = read_excel(main_file_name, sheet_name = assembled_joint_masses)
df_joint = read_excel(main_file_name, sheet_name = joint_coordinates)
df_frame = read_excel(main_file_name, sheet_name = connectivity_frame)
df_restrain = read_excel(main_file_name, sheet_name = restrained)
df_frame_section = read_excel(main_file_name, sheet_name = frame_section)
df_section_properties = read_excel(main_file_name, sheet_name = section_properties)
df_section_rebar = read_excel(main_file_name, sheet_name = section_rebar)
# print(df_main.head()) # shows headers with top 5 rows
# print(df_main.info())

df_section_rebar.columns = df_section_rebar.iloc[0] 
df_section_rebar = df_section_rebar[2:]
df_section_rebar.reset_index(inplace = True, drop = True)

long_dia = df_section_rebar["BarSizeL"]
len_long_dia = len(long_dia)
long_diameter = []

stirrup_dia = df_section_rebar["BarSizeC"]

df_frame.columns = df_frame.iloc[0] 
df_frame = df_frame[2:]
df_frame.reset_index(inplace = True, drop = True)

df_restrain.columns = df_restrain.iloc[0] 
df_restrain = df_restrain[2:]
df_restrain.reset_index(inplace = True, drop = True)

df_frame_section.columns = df_frame_section.iloc[0] 
df_frame_section = df_frame_section[2:]
df_frame_section.reset_index(inplace = True, drop = True)

df_section_properties.columns = df_section_properties.iloc[0] 
df_section_properties = df_section_properties[2:]
df_section_properties.reset_index(inplace = True, drop = True)

df_joint= df_joint.drop(df_joint.index[[1]])
df_joint.reset_index(inplace = True, drop = True)
df_joint.columns = df_joint.iloc[0] 
df_joint = df_joint[1:]

total_joint_rows = len(df_joint.axes[0])
total_frame_rows = len(df_frame.axes[0]) 
total_restrain_rows = len(df_restrain.axes[0]) 

df_joint_mass= df_joint_mass.drop(df_joint_mass.index[[1]])
df_joint_mass.reset_index(inplace = True, drop = True)
df_joint_mass.columns = df_joint_mass.iloc[0] 
df_joint_mass = df_joint_mass[1:]

stirrup_diameter = []

i = 1
while i <= len_long_dia:
    diameter_long = long_dia.iloc[i-1]
    diameter_long = diameter_long.replace("d", " ")
    long_diameter.append(diameter_long)
    
    diameter_stirrup = stirrup_dia.iloc[i-1]
    diameter_stirrup = diameter_stirrup.replace("d", " ")
    stirrup_diameter.append(diameter_stirrup)

    i = i+1
    
long_diameter = pd.DataFrame(long_diameter)
long_dia = long_diameter.select_dtypes(include='object').columns
long_diameter[long_dia] = long_diameter[long_dia].astype("int")  

stirrup_diameter = pd.DataFrame(stirrup_diameter)
str_dia = stirrup_diameter.select_dtypes(include='object').columns
stirrup_diameter[str_dia] = stirrup_diameter[str_dia].astype("int")   

df_section_rebar['BarSizeL'] = long_diameter[0].values
df_section_rebar['BarSizeC'] = stirrup_diameter[0].values

# st.sidebar.header("Geometry of Structure")
# numBayX = st.sidebar.number_input("Number of Bay - X: ", value=1, step=1)
# numBayY = st.sidebar.number_input("Number of Bay - Y: ", value=1, step=1) 
# numFloor = st.sidebar.number_input("Number of Floor: ", value=3, step=1)  
# bayWidthX = st.sidebar.number_input("Bay Width - X: ", value=1, step=1)
# bayWidthY = st.sidebar.number_input("Bay Width - Y: ", value=1, step=1)
# storyHeight = st.sidebar.number_input("Story Heights - X: ", value=3.0, step=1.0)
# E = st.sidebar.number_input("Modulus of Elasticity: ", value=28000000., step=1000000.)
# massX = st.sidebar.number_input("Typical Mass for each joint: ", value=10, step=1)


design_type = st.sidebar.selectbox("Design Type: ", {"Elastic Design", "Nonlinear Design"})

if design_type == "Elastic Design":
    switch = 3
elif design_type == "Nonlinear Design":
    switch = 2
    
# set some properties
ops.wipe()

ops.model('Basic', '-ndm', 3, '-ndf', 6)

# properties
# units kN, m

section_tag = 1
secTag_1 = 1001
beamIntTag = 1001
secBeamTag = 2001

# Define materials for nonlinear columns
# ------------------------------------------
# CONCRETE                  tag   f'c        ec0   ecu E
# Core concrete (confined)

# # Cover concrete (unconfined)
# ops.uniaxialMaterial('Concrete04',2, -25000.,  -0.002,  -0.004,  28000000, 0.0, 0.0, 0,1)
# Propiedades de los materiales
fy = 4200000           #Fluencia del acero
Es = 200000000.0      #Módulo de elasticidad del acero
fc = 20000 # kg/cm2             #Resistencia a la compresión del concreto
E  = 28000000  #Módulo de elasticidad del concreto
G  = 0.5*E/(1+0.2)            #Módulo de corte del concreto

cover = 0.04                  #Recubrimiento de vigas y columnas
# Parametros no lineales de comportamiento del concreto
fc1 = -fc                     #Resistencia a la compresión del concreto
Ec1 = E                       #Módulo de elasticidad del concreto
nuc1 = 0.2                    #Coeficiente de Poisson
Gc1 = Ec1/(2*(1+nuc1))        #Módulo de corte del concreto

# Concreto confinado
Kfc = 1.0 # 1.3               # ratio of confined to unconfined concrete strength
Kres = 0.2                    # ratio of residual/ultimate to maximum stress
fpc1 = Kfc*fc1
epsc01 = 2*fpc1/Ec1 
fpcu1 = Kres*fpc1
epsU1 = 5*epsc01#20
lambda1 = 0.1
# Concreto no confinado
fpc2 = fc1
epsc02 = -0.003
fpcu2 = Kres*fpc2
epsU2 = -0.006#-0.01
# Propiedades de resistencia a la tracción
ft1 = -0.14*fpc1
ft2 = -0.14*fpc2
Ets = ft2/0.002
#print(E/10**8, Ets/10**8)

# Concreto confinado          tag  f'c   ec0     f'cu   ecu
# ops.uniaxialMaterial('Concrete02', 1, fpc1, epsc01, fpcu1, epsU1, lambda1, ft1, Ets)
ops.uniaxialMaterial('Concrete02', 1, fpc1, epsc01, fpcu1, epsc01, lambda1, ft1, Ets)
# Concreto no confinado
ops.uniaxialMaterial('Concrete02', 2, fpc1, epsc01, fpcu1, epsc01, lambda1, ft1, Ets)
# ops.uniaxialMaterial('Concrete02', 2, fpc2, epsc02, fpcu2, epsU2, lambda1, ft2, Ets)
# Acero de refuerzo       tag  fy  E0  b
ops.uniaxialMaterial('Steel02', 3, fy, Es, 0.01, 18,0.925,0.15)

# fc0 = -25000.
# fcc = -28000.
# ecc = 0.002
# Ec = 28000000.
# sqrttool = sqrt(float(-fc0))
# Ec = 5000*sqrttool
# E = 28000000.
# G = 11666667
M = 0.

# ops.uniaxialMaterial('Concrete04',1, int(-25000.), float(-0.002),  -0.02,  int(28000000), 0.0, 0.0, 0.1)

# # Cover concrete (unconfined)
# ops.uniaxialMaterial('Concrete04',2, -25000.,  -0.002,  -0.004,  28000000, 0.0, 0.0, 0,1)


# # STEEL
# # Reinforcing steel 
# Ey = 200000000.0    # Young's modulus
# by = 0.01
# R0 = 15.0
# cR1 = 0.925
# cR2 = 0.15
# fy = 420000
# #                        tag  fy E0    b
# ops.uniaxialMaterial('Steel01', 3, int(fy), Ey, by)

coordTransf = "PDelta"
coordTransf1 = "Linear"  # Linear, PDelta, Corotational
coordTransf2 = "Linear"
massType = "-lMass"  # -lMass, -cMass

# add column element
ops.geomTransf(coordTransf, 1, 1, 0, 0)  
ops.geomTransf(coordTransf1, 2, 0, 0, 1)
ops.geomTransf(coordTransf2, 3, 0, 0, 1)

# ops.geomTransf(coordTransf, 1, 0, 0, 1)
# ops.geomTransf(coordTransf1, 2, 1, 0, 0)
# ops.geomTransf(coordTransf2, 3, 1, 0, 0)

# nodeTag = 1

startJointIndex = 1
total_mass = 0
joint_massX = df_joint_mass.U1
joint_massY = df_joint_mass.U2
joint_massZ = df_joint_mass.U3
joint = df_joint.Joint
x_coor = df_joint.XorR
y_coor = df_joint.Y
z_coor = df_joint.Z
deneme = []
while startJointIndex <= total_joint_rows:
    ops.node(int(joint[startJointIndex]), x_coor[startJointIndex], y_coor[startJointIndex], z_coor[startJointIndex])
    ops.mass(int(joint[startJointIndex]), joint_massX[startJointIndex], joint_massY[startJointIndex], joint_massZ[startJointIndex], 1.0e-10, 1.0e-10, 1.0e-10)
    total_mass = total_mass + joint_massX[startJointIndex] 
    startJointIndex +=1


startRestrainIndex = 0
restrain = df_restrain.Joint
while startRestrainIndex <= total_restrain_rows-1:
    ops.fix(restrain[startRestrainIndex], 1, 1, 1, 1, 1, 1)
    startRestrainIndex += 1

frame = df_frame.Frame
joint_I = df_frame.JointI
joint_J = df_frame.JointJ
frame_section = df_frame_section.AnalSect
startFrameIndex = 0   
Area = df_section_properties["Area"].tolist()
TorsConst = df_section_properties["TorsConst"].tolist()
I33 = df_section_properties["I33"].tolist()
I22 = df_section_properties["I22"].tolist()
t3 = df_section_properties.t3
t2 = df_section_properties.t2
section_cover = df_section_rebar.Cover
nol = df_section_rebar.NumBars2Dir
number_of_top = df_section_rebar.NumBars3Dir
number_of_bottom = df_section_rebar.NumBars3Dir
long_bar = df_section_rebar.BarSizeL
width_total = []
depth_total = []
section_type1 = []
frame_list = []
beam_depth = []
beam_width = []
while startFrameIndex <= total_frame_rows-1:
    frame1 = frame[startFrameIndex]
    frame_index = df_frame["Frame"].tolist().index(frame[startFrameIndex])
    analysis_section = frame_section[frame_index]
    analysis_index = df_section_properties["SectionName"].tolist().index(frame_section[frame_index])

    joint1 = joint_I[startFrameIndex]
    joint2 = joint_J[startFrameIndex]
    jointI_index = df_joint["Joint"].tolist().index(joint_I[startFrameIndex])
    jointJ_index = df_joint["Joint"].tolist().index(joint_J[startFrameIndex])
    z_coordinate_I = z_coor[jointI_index+1]
    z_coordinate_J = z_coor[jointJ_index+1]
    y_coordinate_I = y_coor[jointI_index+1]
    y_coordinate_J = y_coor[jointJ_index+1]
    x_coordinate_I = x_coor[jointI_index+1]
    x_coordinate_J = x_coor[jointJ_index+1]
    

    if z_coordinate_I == z_coordinate_J:
        
        section_type1.append("Beam")
        
        pi = 3.141593;
        dia1 = long_bar[analysis_index]/1000
        As = dia1**2*pi/4;     # area of no. 7 bars
        
        width = t3[analysis_index]
        depth = t2[analysis_index]  
        width_total.append(width)
        depth_total.append(depth)
        frame_list.append(frame[startFrameIndex])
        cover = section_cover[analysis_index]
        number_of_layer = nol[analysis_index]
        n_top = number_of_top[analysis_index]
        n_bot = number_of_bottom[analysis_index]
        n_int = 2
        
        b1 = width/2 - cover
        b2 = (width/2 - cover)*-1
        h1 = depth/2 - cover
        h2 = (depth/2 - cover)*-1
        k_1 = 1/3-0.21*width/depth*(1-(width/depth)**4/12)
        Jc = k_1*width**3*depth
        
        # some variables derived from the parameters
        y1 = depth/2.0
        z1 = width/2.0
        total_y = depth - 2*cover
        total_y_layer = total_y/(number_of_layer-1)
        total_y_layer_step = total_y/(number_of_layer-1)

        
        ops.section('Elastic', secTag_1, E, Area[analysis_index], I33[analysis_index], I22[analysis_index], G, TorsConst[analysis_index])

        ops.section('Fiber', section_tag, '-GJ', G*Jc)
        
        # Create the concrete core fibers
        ops.patch('rect',2,50,1 ,cover-y1, cover-z1, y1-cover, z1-cover)
                
        # Create the concrete cover fibers (top, bottom, left, right)
        ops.patch('rect',2,50,1 ,-y1, z1-cover, y1, z1)
        ops.patch('rect',2,50,1 ,-y1, -z1, y1, cover-z1)
        ops.patch('rect',2,2,1 ,-y1, cover-z1, cover-y1, z1-cover)
        ops.patch('rect',2,2,1 , y1-cover, cover-z1, y1, z1-cover)
        
        top = ['layer','straight', 3, n_top, As, y1-cover-dia1, cover-z1+dia1, y1-cover-dia1, z1-cover-dia1]
        bottom = ['layer','straight', 3, n_bot, As, cover-y1+dia1, cover-z1+dia1, cover-y1+dia1, z1-cover-dia1]
        
        fib_sec_2 = [['section', 'Fiber', 1],
        ['patch', 'rect',2,50,1 ,-y1, z1-cover, y1, z1],
        ['patch', 'rect',2,50,1 ,-y1, -z1, y1, cover-z1],
        ['patch', 'rect',2,2,1 ,-y1, cover-z1, cover-y1, z1-cover],
        ['patch', 'rect',2,2,1 , y1-cover, cover-z1, y1, z1-cover],
        ['patch', 'rect',1,50,1 ,cover-y1, cover-z1, y1-cover, z1-cover],
        top,
        bottom]
        
        ops.layer('straight', 3, n_top, As, y1-cover, cover-z1, y1-cover, z1-cover)
        ops.layer('straight', 3, n_bot, As, cover-y1, cover-z1, cover-y1, z1-cover)
        
        total_int_layer = number_of_layer-2
        int_layer = 1
        
        ops.beamIntegration("Lobatto", beamIntTag,section_tag,5)
        
        while int_layer <= total_int_layer:
        
            ops.layer('straight', 3, n_int, As, y1-cover-total_y_layer, cover-z1+dia1, y1-cover-total_y_layer, z1-cover-dia1)
            int_layer_def = ['layer','straight', 3, n_int, As, y1-cover-total_y_layer, cover-z1+dia1, y1-cover-total_y_layer, z1-cover-dia1]
            fib_sec_2.append(int_layer_def)
            total_y_layer = total_y_layer + total_y_layer_step
            int_layer = int_layer +1
            
        matcolor = ['r', 'lightgrey', 'gold', 'w', 'w', 'w']
        # opsv.plot_fiber_section(fib_sec_2, matcolor=matcolor)
        plt.axis('equal')    
        numIntgrPts = 5
        
        if y_coordinate_I == y_coordinate_J:
            if switch == 1:
                ops.element('forceBeamColumn', int(frame[startFrameIndex]), int(joint_I[startFrameIndex]), int(joint_J[startFrameIndex]), 2, beamIntTag)
            elif switch == 2:
                ops.element('nonlinearBeamColumn', int(frame[startFrameIndex]), int(joint_I[startFrameIndex]), int(joint_J[startFrameIndex]), numIntgrPts, section_tag, 2, '-integration', 'Lobatto')
            elif switch == 3:
                ops.element('elasticBeamColumn', int(frame[startFrameIndex]), int(joint_I[startFrameIndex]), int(joint_J[startFrameIndex]), Area[analysis_index], E, G, TorsConst[analysis_index], I33[analysis_index], I22[analysis_index], 2, '-mass', M, massType)
        else:
            if switch == 1:
                ops.element('forceBeamColumn', int(frame[startFrameIndex]), int(joint_I[startFrameIndex]), int(joint_J[startFrameIndex]), 3, beamIntTag)
            elif switch == 2:
                ops.element('nonlinearBeamColumn', int(frame[startFrameIndex]), int(joint_I[startFrameIndex]), int(joint_J[startFrameIndex]), numIntgrPts, section_tag, 3, '-integration', 'Lobatto')
            elif switch == 3:
                ops.element('elasticBeamColumn', int(frame[startFrameIndex]), int(joint_I[startFrameIndex]), int(joint_J[startFrameIndex]), Area[analysis_index], E, G, TorsConst[analysis_index], I33[analysis_index], I22[analysis_index], 3, '-mass', M, massType)
    else:
        
        pi = 3.141593;
        dia1 = long_bar[analysis_index]/1000
        As = dia1**2*pi/4;     # area of no. 7 bars
        section_type1.append("Column")
        width = t2[analysis_index]
        depth = t3[analysis_index]
        width_total.append(width)
        depth_total.append(depth)
        frame_list.append(frame[startFrameIndex])
        cover = section_cover[analysis_index]
        number_of_layer = nol[analysis_index]
        n_top = number_of_top[analysis_index]
        n_bot = number_of_bottom[analysis_index]
        n_int = 2
        
        b1 = width/2 - cover
        b2 = (width/2 - cover)*-1
        h1 = depth/2 - cover
        h2 = (depth/2 - cover)*-1
        k_1 = 1/3-0.21*width/depth*(1-(width/depth)**4/12)
        Jc = k_1*width**3*depth
        
        # some variables derived from the parameters
        y1 = depth/2.0
        z1 = width/2.0
        total_y = depth - 2*cover
        total_y_layer = total_y/(number_of_layer-1)
        total_y_layer_step = total_y/(number_of_layer-1)
        
        ops.section('Elastic', secTag_1, E, Area[analysis_index], I33[analysis_index], I22[analysis_index], G, TorsConst[analysis_index])
                
        ops.section('Fiber', section_tag, '-GJ', G*Jc)
        
        # Create the concrete core fibers
        ops.patch('rect',1,50,1 ,cover-y1, cover-z1, y1-cover, z1-cover)
        
        
        # Create the concrete cover fibers (top, bottom, left, right)
        ops.patch('rect',2,50,1 ,-y1, z1-cover, y1, z1)
        ops.patch('rect',2,50,1 ,-y1, -z1, y1, cover-z1)
        ops.patch('rect',2,2,1 ,-y1, cover-z1, cover-y1, z1-cover)
        ops.patch('rect',2,2,1 , y1-cover, cover-z1, y1, z1-cover)
        
        top = ['layer','straight', 3, n_top, As, y1-cover-dia1, cover-z1+dia1, y1-cover-dia1, z1-cover-dia1]
        bottom = ['layer','straight', 3, n_bot, As, cover-y1+dia1, cover-z1+dia1, cover-y1+dia1, z1-cover-dia1]
        
        fib_sec_2 = [['section', 'Fiber', 1],
        ['patch', 'rect',2,50,1 ,-y1, z1-cover, y1, z1],
        ['patch', 'rect',2,50,1 ,-y1, -z1, y1, cover-z1],
        ['patch', 'rect',2,2,1 ,-y1, cover-z1, cover-y1, z1-cover],
        ['patch', 'rect',2,2,1 , y1-cover, cover-z1, y1, z1-cover],
        ['patch', 'rect',1,50,1 ,cover-y1, cover-z1, y1-cover, z1-cover],
        top,
        bottom]
        
        ops.layer('straight', 3, n_top, As, y1-cover, cover-z1, y1-cover, z1-cover)
        ops.layer('straight', 3, n_bot, As, cover-y1, cover-z1, cover-y1, z1-cover)
        
        total_int_layer = number_of_layer-2
        int_layer = 1
        
        ops.beamIntegration("Lobatto", beamIntTag,section_tag,5)
        
        while int_layer <= total_int_layer:
        
            ops.layer('straight', 3, n_int, As, y1-cover-total_y_layer, cover-z1+dia1, y1-cover-total_y_layer, z1-cover-dia1)
            int_layer_def = ['layer','straight', 3, n_int, As, y1-cover-total_y_layer, cover-z1+dia1, y1-cover-total_y_layer, z1-cover-dia1]
            fib_sec_2.append(int_layer_def)
            total_y_layer = total_y_layer + total_y_layer_step
            int_layer = int_layer +1
            
        matcolor = ['r', 'lightgrey', 'gold', 'w', 'w', 'w']
        # opsv.plot_fiber_section(fib_sec_2, matcolor=matcolor)
        plt.axis('equal')  
        
        numIntgrPts = 8
        if switch == 1:
            ops.element('forceBeamColumn', int(frame[startFrameIndex]), int(joint_I[startFrameIndex]), int(joint_J[startFrameIndex]), 1, beamIntTag)
        elif switch == 2:
            ops.element('nonlinearBeamColumn', int(frame[startFrameIndex]), int(joint_I[startFrameIndex]), int(joint_J[startFrameIndex]), numIntgrPts, section_tag, 1, '-integration', 'Lobatto')
        elif switch == 3:
            ops.element('elasticBeamColumn', int(frame[startFrameIndex]), int(joint_I[startFrameIndex]), int(joint_J[startFrameIndex]), Area[analysis_index], E, G, TorsConst[analysis_index], I33[analysis_index], I22[analysis_index], 1, '-mass', M, massType)
    startFrameIndex +=1
    section_tag += 1
    secTag_1 += 1
    beamIntTag += 1
    secBeamTag += 1

# calculate eigenvalues & print results
numEigen = 3
eigenValues = ops.eigen(numEigen)
PI = 2 * asin(1.0)

period_list = []
for i in range(0, numEigen):
    lamb = eigenValues[i]
    period = 2 * PI / sqrt(lamb)
    period_list.append(period)

        
###################################
#### Display the active model with node tags only
opsplt.plot_model("nodes")

####  Display specific mode shape with scale factor of 300 using the active model
opsplt.plot_modeshape(1, 50)
####  Display specific mode shape with scale factor of 300 using the active model
opsplt.plot_modeshape(2, 50)
####  Display specific mode shape with scale factor of 300 using the active model
opsplt.plot_modeshape(3, 50)

###################################
# To save the analysis output for deformed shape, use createODB command before running the analysis
# The following command saves the model data, and output for gravity analysis and the first 3 modes 
# in a folder "3DFrame_ODB"

# opsplt.createODB("3DFrame", "Gravity", Nmodes=3)


# # # Define Static Analysis
# ops.timeSeries('Linear', 1)
# ops.pattern('Plain', 1, 1)
# ops.load(total_joint_rows, 3, 0, 0, 0, 0, 0)
# ops.constraints('Transformation')
# ops.numberer('RCM')
# ops.system('BandGeneral')
# ops.test('NormDispIncr', 1.0e-6, 6, 2)
# ops.algorithm('Linear')
# ops.integrator('LoadControl', 1)
# ops.analysis('Static')

# # # Run Analysis
# ops.analyze(10)

# for i in range(0, numEigen):
#     lamb = eigenValues[i]
#     period = 2 * PI / sqrt(lamb)

# # IMPORTANT: Make sure to issue a wipe() command to close all the recorders. Not issuing a wipe() command
# # ... can cause errors in the plot_deformedshape() command.

# ops.wipe()

# ####################################
# ### Now plot mode shape 2 with scale factor of 300 and the deformed shape using the recorded output data

# opsplt.plot_modeshape(2, 300, Model="3DFrame")
# opsplt.plot_deformedshape(Model="3DFrame", LoadCase="Gravity")

# st.success("Analysis Compeleted!")
# st.info("First Period of System is: " + str(format(period, ".2f")) + " sec")
# st.markdown("First Period of System is:" + str(format(period, ".2f")) + " sec")

fig_wi_he = 30., 20.
ele_shapes = []
startFrameIndex = 0
while startFrameIndex <= total_frame_rows-1:
    x = ['rect', [width_total[startFrameIndex], depth_total[startFrameIndex]]]
    ele_shapes.append(x)   
    startFrameIndex = startFrameIndex +1
    
fruit_dictionary = dict(zip(frame_list, ele_shapes))
deneme3 = opsv.plot_extruded_shapes_3d(fruit_dictionary, fig_wi_he=fig_wi_he)

plt.show()
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot(deneme3)


