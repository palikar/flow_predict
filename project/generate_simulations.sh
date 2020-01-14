#!/usr/bin/env sh



for res_dir in $(ls -1v -d1 results_c/plain_results_*); do

    if [ ! -d "${res_dir}"/simulations/full_simulation ]; then
        continue
    fi

        
    for rec_dir in $(ls -1v -d1 "${res_dir}"/recursive_i*) ; do
        ./make_anim.py "$rec_dir/x_recursive.gif" $(ls -1v $rec_dir/recursive/x_*)
        ./make_anim.py "$rec_dir/y_recursive.gif" $(ls -1v $rec_dir/recursive/y_*)
    done

    
    
    for sim_dir in $(ls -1v -d1 "${res_dir}"/simulations/full_simulation/simulation_*) ; do
        ./make_anim.py "$sim_dir/x_simulation.gif" $( ls -1v $sim_dir/x_step_*)
        ./make_anim.py "$sim_dir/y_simulation.gif" $( ls -1v $sim_dir/y_step_*)
        [ -f "$sim_dir/p_step_1.png" ] &&  ./make_anim.py "$sim_dir/p_simulation.gif" $( ls -1v $sim_dir/p_step_*)
    done    

done

