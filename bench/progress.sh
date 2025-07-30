# Create a nice formatted summary
printf "%-30s %-10s %-15s\n" "MODEL" "SAMPLES" "SEED"
printf "%-30s %-10s %-15s\n" "-----" "-------" "----"
ls results/ | sed 's/scale_all_subjects_[0-9]\+_[0-9]\+_//g' | \
awk -F'_' '{
    model = $1; 
    for(i=2; i<=NF-3; i++) model = model "_" $i;
    samples = $(NF-2);
    seed = $NF;
    printf "%-30s %-10s %-15s\n", model, samples, seed
}' | sort

