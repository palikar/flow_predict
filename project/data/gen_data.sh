#!/bin/bash

PARAMS_FILE=$(realpath $1)
CONFIG_FILE=$(realpath $2)
GEOMETRY_DATA="$(realpath $3)/"
ROOT_DATA_DIR=$(realpath $4)

# FLOW_EXEC="/home/arnaud/code_sys/Hiflow3_2.0/build/examples/flow/flow_tutorial"
FLOW_EXEC=${FLOW_EXEC:-"/home/arnaud/code_sys/Hiflow3_2.0/build/examples/flow/flow_tutorial"}
MPI_NP=${MPI_NP:-4}



set_params_f()
{
    sed -i -e "s#<Viscosity>.*</Viscosity>#<Viscosity>$1</Viscosity>#g" ${CONFIG_FILE}
    sed -i -e "s#<Density>.*</Density>#<Density>$2</Density>#g" ${CONFIG_FILE}
    sed -i -e "s#<InflowSpeed>.*</InflowSpeed>#<InflowSpeed>$3</InflowSpeed>#g" ${CONFIG_FILE}
}


# ec="echo"
# eval "${ec} 213"

echo "Hiflow executable: " $FLOW_EXEC
echo "MPI workers count:" $MPI_NP
mkdir -p ${ROOT_DATA_DIR}

count=1
while IFS=" " read -r visc dens inflow prefix
do
    output=${prefix}_$count
    sed -i -e "s#<OutputPrefix>.*</OutputPrefix>#<OutputPrefix>${output}</OutputPrefix>#g" ${CONFIG_FILE}
    echo "Running simulation: ${output}"
    set_params_f $visc $dens $inflow
    mkdir -p "${ROOT_DATA_DIR}/${output}"
    cd "${ROOT_DATA_DIR}/${output}"
    mpirun -np ${MPI_NP} ${FLOW_EXEC} "${CONFIG_FILE}" "${GEOMETRY_DATA}"
    cd - > /dev/null
    count=`expr $count + 1`
done < <(tail -n +2 ${PARAMS_FILE} | sed -e 's/\,/ /g') 
