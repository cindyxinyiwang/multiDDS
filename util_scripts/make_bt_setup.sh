
#bt_model=checkpoints/bt_iwslt_deen70k/ende/checkpoint_best.pt
#cp_base_dir=checkpoints/bt_iwslt_deen70k/deen_
#script_base_dir=job_scripts/bt_iwslt_deen70k/deen_
#
#template_dir=util_scripts/bt_dds_templates/
#SEED=2
#
#for dis in 0.90 0.92 0.94 0.96; do
#    for lr in 1e-9; do
#        for temp in dds_m_dbase_dis; do
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

#bt_model=checkpoints/bt_iwslt_deen30k/ende/checkpoint_best.pt
#cp_base_dir=checkpoints/bt_iwslt_deen30k/deen_
#script_base_dir=job_scripts/bt_iwslt_deen30k/deen_
#
#template_dir=util_scripts/bt_dds_templates/
#SEED=2
#
#for dis in 0.90 0.95 0.99; do
#    for lr in 1e-9; do
#    for up in 1; do
#        for temp in dds_m_dbase_dis; do
#            script_name="$script_base_dir""$temp""$dis"_lr"$lr"_up"$up".sh
#            dirname="$cp_base_dir""$temp""$dis"_lr"$lr"_up"$up"
#            echo $script_name
#            echo $dirname
#            sed "s@DIR_NAME@$dirname@g; s@SEED@$SEED@g; s@LR@$lr@g; s@DIS@$dis@g; s@UP@$up@g" < $template_dir/$temp > $script_name 
#            #sed "s/SEED/$SEED/g; s/LR/$lr/g; s/DIS/$dis/g" < $template_dir/$temp > $script_name 
#            mkdir -p $dirname
#            cp $bt_model $dirname/checkpoint_last.pt
#        done
#    done
#    done
#done

bt_model=checkpoints/bt_glgpor/eng_glg/checkpoint_best.pt
cp_base_dir=checkpoints/bt_glgpor/
script_base_dir=job_scripts/bt_glgpor/

template_dir=util_scripts/bt_dds_templates/
SEED=2

for dis in 0.99 0.95 0.90; do
    for lr in 1e-9; do
    for up in 1 0; do
        for temp in glg_dds_m_dbase_dis; do
            script_name="$script_base_dir""$temp""$dis"_lr"$lr"_up"$up".sh
            dirname="$cp_base_dir""$temp""$dis"_lr"$lr"_up"$up"
            echo $script_name
            echo $dirname
            sed "s@DIR_NAME@$dirname@g; s@SEED@$SEED@g; s@LR@$lr@g; s@DIS@$dis@g; s@UP@$up@g" < $template_dir/$temp > $script_name 
            #sed "s/SEED/$SEED/g; s/LR/$lr/g; s/DIS/$dis/g" < $template_dir/$temp > $script_name 
            mkdir -p $dirname
            cp $bt_model $dirname/checkpoint_last.pt
        done
    done
    done
done
