import numpy as np
import glob
import sys

def main():
    print('test')

def help():
    help_s = '''
        usage: pipeline.py living_plants_path flushed_plants_path [options] ...
    
        This is a tool to detect and analyze the ratio between embolism regions in living and flushed plants. The tool requires seperate paths to the top directories containing all the living and flushed plant images
        
        living_plants_path      A path to the top directory of all living plants images.
        flushed_plants_path     A path to the top directory of all flushed plants images.
        **Note**: By default the tool will analyze all .TIF files in the subdirectories from the paths, but the type could be modified by [-type <type>]
        
        Options:
            -h, -help, --h, --help  Displays this page. Shows instructions on how to use the tool and its different options.
            -model <models_path>    Modifies the location of the models. By defult models folder is expected at .\models. 
                                    The names of the model files need to be model._l.pickle and model_f.pickle for the living plants and flushed plants respectively.
            -type <img_type>        Modifies type of the images used. By defult the type is TIF.
        
        Example:
        \tpipeline.py living_plants_path flushed_plants_path -model ./models/ -type PNG

        '''
    print(help_s)

if __name__=='__main__':
    # command line options
    if np.any(sys.argv == '-model'): # check if model option used
        models_path = sys.argv[np.argmax(np.array(sys.argv,dtype=np.object)=='-model')+1]
    if np.any(sys.argv == '-type'):# check if type option used
        img_type = sys.argv[np.argmax(np.array(sys.argv,dtype=np.object)=='-type')+1]    
    if np.any(sys.argv == ['-h','-help','--h','--help']) or len(sys.argv)<3:
        help()
    else:
        path_l = sys.argv[1]
        path_f = sys.argv[2]
        main()