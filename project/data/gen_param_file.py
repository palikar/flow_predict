#!/usr/bin/env python
import os
import sys
import itertools


##########################################################
#  ____                                _                 #
# |  _ \ __ _ _ __ __ _ _ __ ___   ___| |_ ___ _ __ ___  #
# | |_) / _` | '__/ _` | '_ ` _ \ / _ \ __/ _ \ '__/ __| #
# |  __/ (_| | | | (_| | | | | | |  __/ ||  __/ |  \__ \ #
# |_|   \__,_|_|  \__,_|_| |_| |_|\___|\__\___|_|  |___/ #
##########################################################

# baseline model
simple_speed = 3.5
simple_viscosity = 0.1
simple_density = 4

# constant parameters to be used when the other ones are varied
const_speed = 0.3
const_viscosity = 0.3
const_density = 1

# viscosity and density model
viscosity_start = 0.015
viscosity_end = 0.03

density_start = 1
density_end = 22

# inflow speed model
speed_start = 0.5
speed_end = 10

# count of the range devisions
inflow_range_divisions = 40
fluid_range_divisions = 20

##########################################################
##########################################################

def main():

    file_name = sys.argv[1]
    print("Generating parameters list in file:", file_name)

    with open(file_name, 'w') as output:
        output.write("viscosity, density, inflow_speed, prefix\n")

        print('Baseline parameters: visc - {}, dens - {}, inflow - {}'.format(simple_viscosity, simple_density, simple_speed))
        output.write("{}, {}, {}, {}\n".format(simple_viscosity, simple_density, simple_speed, "c"))

        l = range(inflow_range_divisions)
        print('Generating inflow speeds:')
        print('    range: [{}, {}]'.format(speed_start, speed_end))
        print('    intervals count: {}'.format(len(l)))
        print('    viscosity: {}'.format(const_viscosity))
        print('    density: {}'.format(const_density))
        
        for i in l:
            speed = speed_start + i*(speed_end - speed_start)/(inflow_range_divisions - 1)
            output.write("{}, {}, {}, {}\n".format(const_viscosity, const_density, speed, "s"))        

        densities = [density_start + i*(density_end - density_start)/(fluid_range_divisions - 1) for i in range(fluid_range_divisions)]
        viscosities = [viscosity_start + i*(viscosity_end - viscosity_start)/(fluid_range_divisions - 1) for i in range(fluid_range_divisions)]
        l = [i for i in itertools.product(viscosities, densities)]
        print('Generating fluid and viscosities:')
        print('    viscosity range: [{}, {}]'.format(viscosity_start, viscosity_end))
        print('    density range: [{}, {}]'.format(density_start, density_end))
        print('    intervals count: {}'.format(len(l)))
        print('    inflow: {}'.format(const_speed))

        for v, d in l:
            output.write("{}, {}, {}, {}\n".format(v, d, const_speed, "vd"))


if __name__ == '__main__':
    main()
