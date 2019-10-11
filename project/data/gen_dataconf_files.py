#!/usr/bin/env python
import os
import sys
import re

def key_func(obj1):
    l = re.findall(r'\d+', obj1)
    return int(l[0])

def main():

    params_file = sys.argv[1]
    root_dir = sys.argv[2]

    prexises = set()
    with open(params_file, 'r') as handle:
        handle.readline()
        prefix = handle.readline().rstrip('\n').split(',')[-1].strip()
        while prefix:
            prexises.add(prefix)
            prefix = handle.readline().rstrip('\n').split(',')[-1].strip()

    print('Prefixes:', ', '.join(prexises))

    for pref in prexises:
        print('Handling prefix: ', pref)

        dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d)) and str(d).startswith(pref)]

        print('Directories with thie prefix:', ', '.join(dirs))

        dataconf_file_name = os.path.join(root_dir, pref + '_dataconf.txt')
        with open(dataconf_file_name, 'w') as config_handle:
            print('Creating dataconf file', dataconf_file_name)
            if pref == 'c':
                config_handle.write('plain\n')

            if pref == 'vd':
                config_handle.write('fluid\n')

            if pref == 's':
                config_handle.write('speed\n')

            for sim_dir in dirs:
                x_img_dir = os.path.join(sim_dir, 'images_x')
                # y_img_dir = os.path.join(sim_dir, 'images_y')
                # ü_img_dir = os.path.join(sim_dir, 'images_p')

                params_file = os.path.join(sim_dir, 'params.txt')

                params = open(params_file, 'r').readline().rstrip('\n').split(',')


                imgs = os.listdir(x_img_dir)
                imgs.sort(key=key_func)

                for a, b in zip(imgs[0:-1],imgs[1:]):
                    a_x = os.path.join(os.path.basename(sim_dir), 'images_x', a)
                    b_x = os.path.join(os.path.basename(sim_dir), 'images_y', b)

                    a_y = os.path.join(os.path.basename(sim_dir), 'images_x', a)
                    b_y = os.path.join(os.path.basename(sim_dir), 'images_y', b)

                    a_p = os.path.join(os.path.basename(sim_dir), 'images_p', a)
                    b_p = os.path.join(os.path.basename(sim_dir), 'images_p', b)

                    if pref == 'c':
                        config_handle.write('{}, {}, {}, {}, {}, {}\n'.format(a_x, b_x, a_p,
                                                                              a_y, b_y, b_p))

                    if pref == 'vd':
                        config_handle.write('{}, {}, {}, {}, {}, {}, {}, {}\n'.format(a_x, b_x, a_p,
                                                                                      a_y, b_y, b_p,
                                                                                      params[0], params[1]))

                    if pref == 's':
                        config_handle.write('{}, {}, {}, {}, {}, {}, {}\n'.format(a_x, b_x, a_p,
                                                                                  a_y, b_y, b_p,
                                                                                  params[2]))


if __name__ == '__main__':
    main()
