
#bt_model=checkpoints/bt_iwslt_deen70k/ende/checkpoint_best.pt
#cp_base_dir=checkpoints/bt_iwslt_deen70k/deen_
#script_base_dir=job_scripts/bt_iwslt_deen70k/deen_
#
#template_dir=util_scripts/bt_dds_templates/
#SEED=2
#
#for dis in 0.99 0.95; do
#    for lr in 1e-10 1e-9; do
#        for temp in dds_dis; do
#            script_name="$script_base_dir""$temp""$dis"_lr"$lr".sh
#            dirname="$cp_base_dir""$temp""$dis"_lr"$lr"
#            echo $script_name
#            echo $dirname
#            sed "s@DIR_NAME@$dirname@g; s@SEED@$SEED@g; s@LR@$lr@g; s@DIS@$dis@g" < $template_dir/$temp > $script_name 
#            #sed "s/SEED/$SEED/g; s/LR/$lr/g; s/DIS/$dis/g" < $template_dir/$temp > $script_name 
#            mkdir -p $dirname
#            cp $bt_model $dirname/checkpoint_last.pt
#        done
#    done
#done

bt_model=checkpoints/bt_azetur/eng_azetur/checkpoint_best.pt
cp_base_dir=checkpoints/bt_azetur/
script_base_dir=job_scripts/bt_azetur/

template_dir=util_scripts/bt_dds_templates/
SEED=2

for dis in 0.99; do
    for lr in 1e-9; do
        for temp in aze_dds_dis; do
            script_name="$script_base_dir""$temp""$dis"_lr"$lr".sh
            dirname="$cp_base_dir""$temp""$dis"_lr"$lr"
            echo $script_name
            echo $dirname
            sed "s@DIR_NAME@$dirname@g; s@SEED@$SEED@g; s@LR@$lr@g; s@DIS@$dis@g" < $template_dir/$temp > $script_name 
            #sed "s/SEED/$SEED/g; s/LR/$lr/g; s/DIS/$dis/g" < $template_dir/$temp > $script_name 
            mkdir -p $dirname
            cp $bt_model $dirname/checkpoint_last.pt
        done
    done
done
