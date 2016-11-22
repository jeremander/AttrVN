
if __name__ == '__main__':
    import pickle
    import os
    import importlib
    import sys
    paramlist = pickle.load(open('tmp/params%d.pickle' % (1- 1), 'rb'))
    suite_bundle = pickle.load(open('tmp/suite.pickle', 'rb'))
    #sys.path.append(os.path.dirname(suite_bundle['module_path_']))
    os.chdir(os.path.dirname(suite_bundle['module_path_']))
    suite_module = importlib.import_module(os.path.basename(suite_bundle['module_path_']).split('.')[0])
    print(suite_module.__dict__)
    class_name_ = suite_bundle['class_name_']
    class_ = getattr(suite_module, class_name_)
    suite = class_()
    suite.options.sge = False
    suite.do_experiment(paramlist)
        