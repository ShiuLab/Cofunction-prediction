################################################################################
#
# By: Shinhan Shiu
# Date: 11/3/21
#
# Purpose: Generate features for Ally's co-function prediction. 
#
# Approach: 
#   For each trio of fitness values (m1, m2, m12), 5 steps:
#
#   1. Transform each value: square, sqrt, log, inverse, also keep the original.
#      At this step: 5 m1s, 5 m2s, 5 m12s.
#   2. Combine m1 & m2: +, -, x, /, min, max, average, call it m1_m2. 
#      At this step: 5 x 7 = 35 m1_m2 combinations.
#   3. Transform m1_m2 values as in 1. 
#      At this step: 35 x 5 = 175 transformed m1_m2 values.
#   4. Combine m1_m2 with m12 to get genetic interaction score (gis).
#      At this step: 175 (m1_m2) x 5 (m12) x 7 = 6125 gis values.
#   5. Transform gis as in 1.
#      At this step: 6125 x 5 = 30625 transformed gis values.
#
#   At the end of the run, each gene pair should have 30625 combinations.
#   Together with the original m1, m2, m12 values, there are 30628 features in
#   the output table.
#
################################################################################
import sys, math, statistics

# Goal: Transform each values with 4 operations: 
#       square (S), sqrt (Q), log (L), inverse (I)
# Arguments:
#   val  - value to transform
#   feat - feature name including info on what has been done to the value
# Return: a nested list with each element specific to a transformation, there
#   are two sub-elements in each element: original/transformed value and feature
#   name.
def transform_each(val, feat="", debug=0):
    v0 = val
    v1 = val**2
    
    if val >0:
        v2 = math.sqrt(val)
    else:
        v2 = math.sqrt(-val)
    if val != 0:
        if val > 0:
            v3 = math.log(val)
        else:
            v3 = math.log(-val)
        v4 = 1/val
    else:
        v3 = v4 = val
    
    t_val = [[v0, feat+"O"], [v1, feat+"S"], [v2, feat+"R"], 
             [v3, feat+"L"], [v4, feat+"I"]]
    if debug: 
        print("  transform:",val)
        print("  transform out:",t_val)
        if feat in ["O", "OpO", "OpOO", "OpOOpO", "OpOOpOO"]:
            print(f">feat={feat}O, v0={v0}")
    
    return t_val

# Goal: Given 2 lists, combine values of the same indicies from the lists (i.e.,
#   1st element in list 1 with 1st element in list 2) then transform each of the
#   combined values again. When the values are being operated on, each
#   combined value will also be assigned a feature name.
# Arguments:
#   m_fit1: For mutant 1, a nested list of 5 transformed values and feat names
#   m_fit2: For mutant 2, same length as m_fit1
# Return:
#   tr_1_2: a nested list with 5 transformations for mutant1 and mutant 2 x 
#     7 ways to combine transformed value pairs x 5 transformation of combined
#     value = 5 x 7 x 5 = 175 values
def combine_m1_m2(m_fit1, m_fit2, debug=0):

    # Iterate through each element of each list and generate seven different
    # combined values, then each is transform into 5 values. So totally 35
    # values for each element pair.
    
    # For holding the final output:
    # tr_1_2 = [[value1, feat1], [value2, feat2],...]
    tr_1_2 = []
    
    if debug:
        print("m_fit1:",m_fit1)
    
    # Iterate through pairs of mutant 1 and 2 fitness values and feat names
    for idx in range(len(m_fit1)):
        v1 = m_fit1[idx][0]
        v2 = m_fit2[idx][0]
        f1 = m_fit1[idx][1]
        f2 = m_fit2[idx][1]
        
        if debug:
            print("  m1_m2:",v1,f1,v2,f2)
            
        # Combine individual fitness values
        # v_1_2: a 7-value list, each different way to combine m1 & m2
        # f_1_2: a 7-value list, each with an abbrev on how m1/m2 are combined
        v_1_2, f_1_2 = combine(v1, v2, f1, f2, debug)            
    
        # Transform each of the combined values again, out put will be:
        for idx2 in range(len(v_1_2)):
            val  = v_1_2[idx2]
            feat = f_1_2[idx2]
            
            if debug:
                if feat == "OpO":
                    print(f"feat={feat}, v0={val}") 
            
            tr_1_2_each =  transform_each(val, feat, debug)
            tr_1_2.extend(tr_1_2_each)
    
    if debug:
        print("  len(tr_1_2)=",len(tr_1_2))
        for i in tr_1_2:
            print("tr_1_2 element:",i)
            
    return tr_1_2

# Goal: similar to combine_m1_m2, but here is about combining the combined m1_m2
#   value with m_12 fitness value in all possible combinations.
# Arguments:
#   m_m1_m2: a nested list of combined fitness values of mutant 1 and 2 and feat
#     names from combine_m1_m2()
#   m_fit12: a nested list of modified double mutant fitness values and feat
#     names from transform_each()
# Return:
#   GIS: a nested list of combined m1_m2 and m12 values and feat names. It
#     should have 175 m1_m2 values x 5 m_fit2 values x 7 combination type x
#     5 transformations = 30625. 
def combine_m_1_m2_m12(m_m1_m2, m_fit12, debug=0):
    GIS = []
    
    if debug:   
        print(f"len(m_m1_m2):{len(m_m1_m2)}")
        print(f"len(m_fit12):{len(m_fit12)}")
        print("m_m1_m2[0]:",m_m1_m2[0])
    
    # Iterate through mutant 1 fitness values and feature names, this should 
    # have 175 elements.
    for [v1,f1] in m_m1_m2:
        # iterate through mutant 2 values to get all pairwise combo
        # This list should have 5 elements.
        for [v2,f2] in m_fit12:
            if debug:   
                if f1 == "OpOO" and f2 == "O":
                    print(f"v1:{v1},f1:{f1},v2:{v2},f2:{f2}")
        
            # Combine values
            # v_gis: a 7-value list, each different way to combine m1_2 and m_12
            # f_gis: a 7-value list, each with an abbrev on the specific combo
            v_gis, f_gis = combine(v1, v2, f1, f2, debug)
        
            # Transform each of the combined values again, out put will be:
            for idx2 in range(len(v_gis)):
                val  = v_gis[idx2]
                feat = f_gis[idx2]               
                gis =  transform_each(val, feat, debug)
                GIS.extend(gis)
                
                if debug:
                    if feat == "OpOOpO":
                        print(f"feat={feat}, v0={val}") 
 
    if debug:
        print("len(GIS)=",len(GIS))
        for i in GIS:
            print("GIS element:",i)
            
    return GIS
    
# Goal: combine modified fitness values from mutant 1 and 2 with 7 operations
# Return: two lists, v_m1_m2, f_m1_m2
#   v_m1_m2 - 7 elements, each combines m1 and m2 with 1 of 7 operands: 
#     + (a), - (m), x (x), / (d), min (i), max (a), average (v)
#     For substraction, does not make sense to have + vs. - simply because
#     of which gene is called gene 1 in a pair. So all substracted values are
#     set to >=0. Same thing for division, all divided values are >= 1. 
#   f_m1_m2 - a list with 7 elements, each with a lowercase abbrivaition
#     connecting the abbreviations of operations on m1 and m2 previously.
def combine(m1, m2, f1, f2, debug=0):
    # value
    if debug:
        print("  combine:",m1,m2,m1+m2)
        
    v_m1_m2 = [m1+m2, max([m1,m2])-min([m1,m2]), 
               m1*m2, max([m1,m2])/min([m1,m2]),
               min([m1,m2]), max([m1,m2]), statistics.mean([m1,m2])]
    # feature name
    f_m1_m2 = [f'{f1}p{f2}', f'{f1}m{f2}', f'{f1}x{f2}', f'{f1}d{f2}',
               f'{f1}i{f2}', f'{f1}a{f2}', f'{f1}v{f2}']
                 
    return v_m1_m2, f_m1_m2

def help():
    print("\nUsage: script_feature_generation.py input_file debug_flag")
    print("  input_file: a four column file WITHOUT header. The four columns")
    print("    contain gene-pair ID, mutant1 fitness, mutant2 fitness, and")
    print("    double mutant fitness.")
    print("  debug_flag (optional): 0 [default]: no printout. 1: with ")
    print("    printout.\n")
    sys.exit(0)
    
#-------------------------------------------------------------------------------
try:
    # input data
    # 4 columns, tab-delimited: gene1-gene2, fit1, fit2, fit12
    input_file = sys.argv[1]
    
    # Open input file
    inp = open(input_file)    
    
    # Get debug flag if exists
    if len(sys.argv) == 3:
        debug = int(sys.argv[2])
    else:
        debug = 0
# No file argument passed leading to IndexError
except IndexError:
    help()
# Input file does not exists
except FileNotFoundError:
    print("\nERROR: The file specified does not exist.")
    help()
# Input file does not exists
except ValueError:
    print("\nERROR: Debug flag should be 0 or 1.")
    help()    

# Setup output
oup = open(input_file + ".features_out", "w")

# Go through each line
inl = inp.readline()   # Read 1st line
out_header = 0         # header for output is written or not
print("Go through gene pairs:")
c = 0
while inl != "":
    try:
        [genepair, fit1, fit2, fit12] = inl.strip().split('\t') # a list of values
    except ValueError:
        print("\nERR: Line does not contain four elements.")
        print("Offending line:", [inl])
        help()
    
    print(c,":", genepair, fit1, fit2, fit12)
    c += 1
    # Modified fitness values
    # original: 0.9, 0.8, 0.1
    if debug:
        print("Transform fitness values")
        
    try:
        m_fit1  = transform_each(float(fit1), "", debug) # 0.9,0.9^2,sqrt(0.9),1/0.9
        m_fit2  = transform_each(float(fit2), "", debug) # 0.8, ....
        m_fit12 = transform_each(float(fit12),"", debug) # 0.1, ....
    except ValueError:
        print("\nERR: Some elements cannot be converted to numbers.")
        print("Offening line elements:",[fit1, fit2, fit12])
        help()
    
    # Combine original/modified mutant 1 and mutant 2 fitness values
    # e.g., m_fit1 + m_fit2
    if debug:
        print("Combine mutant 1 and 2 fitness values")
    m_m1_m2 = combine_m1_m2(m_fit1, m_fit2, debug)
    
    # Combine combined mutant 1 and 2 fitness value with double mutant value
    #print("Combine with double mutant value")
    m_gis = combine_m_1_m2_m12(m_m1_m2, m_fit12, debug)
    
    # Type the m_gis values from float to string and put the feat name in a list
    m_gis2 = []
    f_gis2 = []
    for idx in range(len(m_gis)):
        [v, f] = m_gis[idx] # value and feat_name
        
        if debug:
            print(f"({v},{f})")
            
        m_gis2.append(str(v))
        f_gis2.append(f)
    
    # Write header if not done already
    if out_header == 0:
        oup.write("Pair\tFit1\tFit2\tFit12\t%s\n" % '\t'.join(f_gis2))
        out_header = 1
    
    # Write all values
    oup.write("%s\t%s\t%s\t%s\t%s\n" % 
        (genepair, fit1, fit2, fit12, "\t".join(m_gis2)))
    
    inl = inp.readline()
    
oup.close()