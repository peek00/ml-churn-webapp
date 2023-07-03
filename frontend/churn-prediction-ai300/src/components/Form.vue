<script setup>

</script>

<template>
  <div class="row vh-100 align-items-center justify-content-center">
    <div class="col-8 bg-light h-75 rounded">
      <div class="row h-100 p-3">
        <div class="col-5 bg-dark rounded text-white ">
          Random Content here
        </div>
        <div class="col-7 rounded px-5">
          <div class="row pb-4">
            <h1>
              Customer Churn Prediction
            </h1>
            <h7>
              Fill in the following details and hit submit to get the prediction!
            </h7>
          </div>
        
          <div class="row">
            <div class="col-4">
              <label for="total_long_distance_fee" class="form-label">Total Long Distance Fee</label>
              <div class="input-group mb-3">
                <span class="input-group-text" id="basic-addon1">$</span>
                <input type="number" class="form-control" id="total_long_distance_fee" name="total_long_distance_fee" >
              </div>
            </div>
            <div class="col-4">
              <label for="total_charges_quarter" class="form-label">Total Charges Quarter</label>
              <div class="input-group mb-3">
                <span class="input-group-text" id="basic-addon1">$</span>
                <input type="number" class="form-control" id="total_charges_quarter" name="total_charges_quarter" >
              </div>
            </div>
          </div>

          <div class="col-6 mb-3">
            <label for="tenure_months" class="form-label">Tenure Months</label>
            <input type="number" class="form-control" id="tenure_months" name="tenure_months">
          </div>

          <div class="col-6 mb-3">
            <label for="num_dependents" class="form-label">Num of Dependents</label>
            <input type="number" class="form-control" id="num_dependents" name="num_dependents">
          </div>

          <div class="mb-3 col-3 form-check">
              <input type="checkbox" class="form-check-input" id="married">
              <label class="form-check-label" for="married" name="married">Happily Married?</label>
            </div>
          <div class="col-6 pb-4">
            Contract Type
            <select class="form-select" name="contract_type">
              <option value="Month-to-Month" selected>Month to Month</option>
              <option value="One Year">One Year</option>
              <option value="Two Year">Two Year</option>
            </select>
          </div>
          <h5> Does the user have the following features?</h5>
            <div class="col-12 pb-4">
              <div class="form-check form-check-inline">
                <input class="form-check-input" type="checkbox" id="has_premium_tech_support" value="option1">
                <label class="form-check-label" for="has_premium_tech_support" name="has_premium_tech_support">Premium Tech Support</label>
              </div>
              <div class="form-check form-check-inline">
                <input class="form-check-input" type="checkbox" id="has_device_protection" value="option2">
                <label class="form-check-label" for="has_device_protection" name="has_device_protection">Device Protection</label>
              </div>
              <div class="form-check form-check-inline">
                <input class="form-check-input" type="checkbox" id="has_online_backup" value="option3" >
                <label class="form-check-label" for="has_online_backup" name="has_online_backup">Online Backup</label>
              </div>
            </div>

            <button type="submit" class="btn btn-primary">Get Prediction</button>
        </div>
      </div>
    </div>
  </div>

</template>

<style scoped>

</style>
