dataset = read.csv('Ads_CTR_Optimisation.csv')

N = 10000
d = 10
numbers_of_selections = integer(d) #Ni
sums_of_rewards = integer(d)       #Ri
ads_selected = integer()
total_reward = 0

for(n in 1:N){
  max_upper_bound = 0
  ad = 0
  for(i in 1:d){
    if(numbers_of_selections[i] >0){
      average_reward = sums_of_rewards[i] / numbers_of_selections[i]
      delta_i = sqrt(3/2 * log(n) / numbers_of_selections[i])
      upper_bound = average_reward + delta_i 
    }else{
      upper_bound = 1e400
    }
    if(upper_bound > max_upper_bound){
      max_upper_bound = upper_bound
      ad = i
    }
  }
  ads_selected = append(ads_selected, ad)
  numbers_of_selections[ad] = numbers_of_selections[ad] + 1
  reward = dataset[n, ad]
  sums_of_rewards[ad] = sums_of_rewards[ad] +reward
  total_reward = total_reward + reward
  
}

hist(ads_selected,
     col = "blue",
     main = "Histogram",
     xlab = "Ads",
     ylab = "Selection") 