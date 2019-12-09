#!/bin/bash

set +e

BASEDIR=$(dirname $0)

SOLUTION_LIST=""
# PVPYTHON_EXE=${PVPYTHON_EXE:-"/home/arnaud/temp/Snap/ParaView-5.2.0-Qt4-OpenGL2-MPI-Linux-64bit/bin/pvpython"}
PVPYTHON_EXE=${PVPYTHON_EXE:-"/home/arnaud/Downloads/ParaView-5.2.0-Qt4-OpenGL2-MPI-Linux-64bit/bin/pvpython"}

CROPPER_SCRIPT="${BASEDIR}/cropper.py"

ROOT_FOLDER=$(realpath $1)
SOLUTION_FOLDER=$2


crop_images()
{
    FOLDER=$1
    echo "Cropping images in ${FOLDER}"
    ls -d $FOLDER/* -1 | xargs -I "{}" python ${CROPPER_SCRIPT} "{}" "{}"
}

# Get the solution files
for sol in $(ls ${ROOT_FOLDER}/${SOLUTION_FOLDER}/${SOLUTION_FOLDER}_*.pvtu | sort --general-numeric-sort -r)
do
    sol="${sol}"
    SOLUTION_LIST="'${sol}', ${SOLUTION_LIST}"
done
# echo "Solution files: ${SOLUTION_LIST}"
sed -e "s#'<file_place>'#$SOLUTION_LIST#g" anim.py > anim_temp.py


# render the U-Direction of velocity
IMAGES_DEST="${ROOT_FOLDER}/${SOLUTION_FOLDER}/images_x"
# echo "Rendering images in $IMAGES_DEST"
mkdir -p ${IMAGES_DEST}
sed -e "s#<output_place>#${IMAGES_DEST}/flow.png#g
        s#<var>#u#g" anim_temp.py > anim_temp_x.py
eval "${PVPYTHON_EXE} anim_temp_x.py"
crop_images ${IMAGES_DEST}


# render the U-Direction of velocity
IMAGES_DEST="${ROOT_FOLDER}/${SOLUTION_FOLDER}/images_y"
echo "Rendering images in $IMAGES_DEST"
mkdir -p ${IMAGES_DEST}
sed -e "s#<output_place>#${IMAGES_DEST}/flow.png#g
        s#<var>#v#g" anim_temp.py > anim_temp_y.py
eval "${PVPYTHON_EXE} anim_temp_y.py"
crop_images ${IMAGES_DEST}


# render the U-Direction of velocity
IMAGES_DEST="${ROOT_FOLDER}/${SOLUTION_FOLDER}/images_p"
# echo "Rendering images in $IMAGES_DEST"
mkdir -p ${IMAGES_DEST}
sed -e "s#<output_place>#${IMAGES_DEST}/flow.png#g
        s#<var>#p#g" anim_temp.py > anim_temp_p.py
eval "${PVPYTHON_EXE} anim_temp_p.py"
crop_images ${IMAGES_DEST}

# # # rm -f anim_temp.py
