# -*- coding: utf-8 -*-
"""
Mode Choice Predictor 1.02
Example Script
"""
import modechoicepredictor as mcp

help(mcp.predict_mini)

tripinfo1=[5,3,10]
tripinfo2=[[1,1,10],[4,2,20],[2,2,40],[7,3,50]]

mc1 = mcp.predict_mini(tripinfo1,printresult=True);
mc2 = mcp.predict_mini(tripinfo2,printresult=True);

tripinfo_macro=[6,4,5,3,2,3] 
tripinfo_micro=[6,4,5,3,2,3,10,3,1] 

mc3=mcp.predict_macro(tripinfo_macro)
mc4=mcp.predict_micro(tripinfo_micro)
print(mc3)
print(mc4)