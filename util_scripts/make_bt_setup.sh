
bt_model=checkpoints/bt_iwslt_ende/deen/checkpoint_best.pt
cp_base_dir=checkpoints/bt_iwslt_ende/ende_
script_base_dir=job_scripts/bt_iwslt_ende/ende_

template_dir=util_scripts/bt_dds_templates/
SEED=2

for dis in 0.90 ; do
    for lr in 1e-9; do
    for gradn in 5; do
    for optim in SGD Adam; do
        for temp in dds_m_dbase_nwords dds_m_dbase; do
            script_name="$script_base_dir""$temp"_dis"$dis"_lr"$lr"_gradn"$gradn"_optim"$optim".sh
            dirname="$cp_base_dir""$temp"_dis"$dis"_lr"$lr"_gradn"$gradn"_optim"$optim"
            echo $script_name
            echo $dirname
            sed "s@DIR_NAME@$dirname@g; s@SEED@$SEED@g; s@LR@$lr@g; s@DIS@$dis@g; s@GRADN@$gradn@g; s@OPTIM@$optim@g" < $template_dir/$temp > $script_name 
            #sed "s/SEED/$SEED/g; s/LR/$lr/g; s/DIS/$dis/g" < $template_dir/$temp > $script_name 
            mkdir -p $dirname
            cp $bt_model $dirname/checkpoint_last.pt
        done
    done
    done
    done
done

