log_dir=/workspace/data/energon_logs/opt1
mkdir -p ${log_dir}
gpu_logs=${log_dir}/gpu_logs
cpu_logs=${log_dir}/cpu_logs
mkdir -p ${gpu_logs}
mkdir -p ${cpu_logs}

num_gpu=4
arg1="--tp ${num_gpu}"

# model_name_arr=("opt-30b" "opt-30b" "opt-30b" "opt-30b")
model_name_arr=("opt-30b")
input_len_arr=(512 512 512 512)
out_len_arr=(64 64 64 64)
# num_gpu_arr=(1 2 4 8)
bs_arr=(1 8 16 24)
num_bs=1
percent="0 100 0 100 0 100"

for(( i=0;i<${#model_name_arr[@]};i++)) do
   bs=${bs_arr[i]};
   input_len=${input_len_arr[i]};
   out_len=${out_len_arr[i]};
   model_name=${model_name_arr[i]};
   ps=${percent// /:}
   model_path=${model_name};
   gpu_log="${gpu_logs}/${model_name}_${input_len}_${out_len}_${bs}_${num_bs}_${ps}_0_${num_gpu}.qdrep";
   cpu_log="${cpu_logs}/${model_name}_${input_len}_${out_len}_${bs}_${num_bs}_${ps}_0_${num_gpu}.txt";
   cpu_log_org="${cpu_log}.org";
   cmd_pre="python "
   exec_path="opt_infer.py"
   args_str=" ${model_path} ${arg1}  --max_tokens ${out_len}";
   cmd="${cmd_pre} ${exec_path} $args_str --log-file ${cpu_log_org}";
   echo $cmd;
   $cmd;
   exec_path="opt_infer_prof.py"
   cmd="nsys profile  -c cudaProfilerApi -f true --stats true  -o ${gpu_log} ${cmd_pre} ${exec_path} $args_str --cpu_log_path ${cpu_log}";
   echo $cmd;
   $cmd;
   echo "done";
done


