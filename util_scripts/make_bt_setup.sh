
bt_model=checkpoints/bt_iwslt_ende/deen/checkpoint_best.pt
cp_base_dir=checkpoints/bt_iwslt_ende/ende_
script_base_dir=job_scripts/bt_iwslt_ende/ende_

template_dir=util_scripts/bt_dds_templates/
SEED=2

for dis in 0.99 ; do
    for lr in 1e-8 1e-9; do
    for gradn in 5; do
    for optim in SGD; do
    for coptim in Adam; do
    for cpsteps in 1000; do
    for cplr in 0.01; do
    for clr in 0.001; do
    for vloss in 0.001; do
        for temp in ac; do
            script_name="$script_base_dir""$temp"_dis"$dis"_lr"$lr"_gradn"$gradn"_optim"$optim"_coptim"$coptim"_cpsteps"$cpsteps"_cplr"$cplr"_clr"$clr"_vloss"$vloss".sh
            dirname="$cp_base_dir""$temp"_dis"$dis"_lr"$lr"_gradn"$gradn"_optim"$optim"_coptim"$coptim"_cpsteps"$cpsteps"_cplr"$cplr"_clr"$clr"_vloss"$vloss"
            echo $script_name
            echo $dirname
            sed "s@DIR_NAME@$dirname@g; s@SEED@$SEED@g; s@COPTIM@$coptim@g; s@CSTEPS@$cpsteps@g; s@CPLR@$cplr@g; s@CLR@$clr@g; s@VLOSS@$vloss@g; s@LR@$lr@g; s@DIS@$dis@g; s@GRADN@$gradn@g; s@OPTIM@$optim@g;" < $template_dir/$temp > $script_name 
            #sed "s/SEED/$SEED/g; s/LR/$lr/g; s/DIS/$dis/g" < $template_dir/$temp > $script_name 
            mkdir -p $dirname
            cp $bt_model $dirname/checkpoint_last.pt
        done
    done
    done
    done
    done
    done
    done
    done
    done
done

