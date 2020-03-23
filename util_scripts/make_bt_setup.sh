
#bt_model=checkpoints/bt_iwslt_ende/deen/checkpoint_best.pt
#cp_base_dir=checkpoints/bt_iwslt_ende/ende_
#script_base_dir=job_scripts/bt_iwslt_ende/ende_
#
#template_dir=util_scripts/bt_dds_templates/
#SEED=2
#
#for dis in 0.99 0.95; do
#    for lr in 1e-9; do
#    for gradn in 5; do
#    for optim in SGD ; do
#    for seed in 5 6 ; do
#        for temp in dds_m_dis; do
#            script_name="$script_base_dir""$temp"_dis"$dis"_lr"$lr"_gradn"$gradn"_optim"$optim"_se"$seed".sh
#            dirname="$cp_base_dir""$temp"_dis"$dis"_lr"$lr"_gradn"$gradn"_optim"$optim"_se"$seed"
#            echo $script_name
#            echo $dirname
#            sed "s@DIR_NAME@$dirname@g; s@SEED@$seed@g; s@LR@$lr@g; s@DIS@$dis@g; s@GRADN@$gradn@g; s@OPTIM@$optim@g" < $template_dir/$temp > $script_name 
#            #sed "s/SEED/$SEED/g; s/LR/$lr/g; s/DIS/$dis/g" < $template_dir/$temp > $script_name 
#            mkdir -p $dirname
#            cp $bt_model $dirname/checkpoint_last.pt
#        done
#    done
#    done
#    done
#    done
#done


bt_model=checkpoints/bt_iwslt_deen30k/ende/checkpoint_best.pt
cp_base_dir=checkpoints/bt_iwslt_deen30k/
script_base_dir=job_scripts/bt_iwslt_deen30k/

template_dir=util_scripts/bt_dds_templates/
SEED=2

for dis in 0.99; do
    for lr in 1e-9; do
    for gradn in 5; do
    for optim in SGD ; do
    for seed in 1 2 3 4 ; do
        for temp in deen_dds_m_dis; do
            script_name="$script_base_dir""$temp"_dis"$dis"_lr"$lr"_gradn"$gradn"_optim"$optim"_se"$seed".sh
            dirname="$cp_base_dir""$temp"_dis"$dis"_lr"$lr"_gradn"$gradn"_optim"$optim"_se"$seed"
            echo $script_name
            echo $dirname
            sed "s@DIR_NAME@$dirname@g; s@SEED@$seed@g; s@LR@$lr@g; s@DIS@$dis@g; s@GRADN@$gradn@g; s@OPTIM@$optim@g" < $template_dir/$temp > $script_name 
            #sed "s/SEED/$SEED/g; s/LR/$lr/g; s/DIS/$dis/g" < $template_dir/$temp > $script_name 
            mkdir -p $dirname
            cp $bt_model $dirname/checkpoint_last.pt
        done
    done
    done
    done
    done
done

#bt_model=checkpoints/bt_iwslt_deen_v2/ende/checkpoint_best.pt
#cp_base_dir=checkpoints/bt_iwslt_deen_v2/
#script_base_dir=job_scripts/bt_iwslt_deen_v2/
#
#template_dir=util_scripts/bt_dds_templates/
#SEED=2
#
#for dis in 0.99; do
#    for lr in 1e-9 1e-10; do
#    for gradn in 0; do
#    for optim in SGD Adam; do
#        for temp in  deen_dds_m_dis; do
#            script_name="$script_base_dir""$temp"_dis"$dis"_lr"$lr"_gradn"$gradn"_optim"$optim".sh
#            dirname="$cp_base_dir""$temp"_dis"$dis"_lr"$lr"_gradn"$gradn"_optim"$optim"
#            echo $script_name
#            echo $dirname
#            sed "s@DIR_NAME@$dirname@g; s@SEED@$SEED@g; s@LR@$lr@g; s@DIS@$dis@g; s@GRADN@$gradn@g; s@OPTIM@$optim@g" < $template_dir/$temp > $script_name 
#            #sed "s/SEED/$SEED/g; s/LR/$lr/g; s/DIS/$dis/g" < $template_dir/$temp > $script_name 
#            mkdir -p $dirname
#            cp $bt_model $dirname/checkpoint_last.pt
#        done
#    done
#    done
#    done
#done
#
