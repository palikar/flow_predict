#!/bin/bash

set +e

BASEDIR=$(dirname $0)

SOLUTION_LIST=""
PVPYTHON_EXE=${PVPYTHON_EXE:-"/usr/bin/python"}

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
for sol in $(ls ${ROOT_FOLDER}/${SOLUTION_FOLDER}/images/ | sort --general-numeric-sort -r)
do
		sol="${ROOT_FOLDER}/${SOLUTION_FOLDER}/${sol}"
		SOLUTION_LIST="'${sol}', ${SOLUTION_LIST}"
done
echo "Solution files: ${SOLUTION_LIST}"
sed -e "s#'<file_place>'#$SOLUTION_LIST#g" anim.py > anim_temp.py



# render the U-Direction of velocity
IMAGES_DEST="${ROOT_FOLDER}/${SOLUTION_FOLDER=}/images_x"
crop_images ${IMAGES_DEST}
echo "Rendering images in $IMAGES_DEST"
mkdir -p ${IMAGES_DEST}
script=$(sed -e "s#<output_place>#${IMAGES_DEST}#g
			 			 		 s#<var>#u#g" anim_temp.py)
eval "${PVPYTHON_EXE} -c '${script}'"


# render the V-Direction of velocity
IMAGES_DEST="${ROOT_FOLDER}/${SOLUTION_FOLDER=}/images_y"
crop_images ${IMAGES_DEST}
echo "Rendering images in ${IMAGES_DEST}"
mkdir -p ${IMAGES_DEST}
script=$(sed -e "s#<output_place>#${IMAGES_DEST}#g
				 			 	 s#<var>#v#g" anim_temp.py)
eval "${PVPYTHON_EXE} -c '${script}'"


# render the pressure
IMAGES_DEST="${ROOT_FOLDER}/${SOLUTION_FOLDER=}/images_p"
crop_images ${IMAGES_DEST}
echo "Rendering images in ${IMAGES_DEST}"
mkdir -p ${IMAGES_DEST}
scirpt=$(sed -e "s#<output_place>#${IMAGES_DEST}#g
						 		 s#<var>#p#g" anim_temp.py)
eval "${PVPYTHON_EXE} -c '${script}'"

# rm -f anim_temp.py
