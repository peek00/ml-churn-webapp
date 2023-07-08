<template>
  <div class="row align-items-center justify-content-center" style="height:90vh">
    <div class="col-8 bg-light h-75 rounded">
      <div class="row h-100 p-3">
        <div class="col-5 bg-dark rounded text-white p-4 ps-5">
          <p class="fw-bold ">
            AI 300
          </p>

          <h3 >
            Predict your customer churn with us now!
          </h3>
          <p>
            This machine learning prediction uses a CatBoost model and Principal Component Analysis (PCA) to predict whether a customer is likely to churn or not.
            ,<br>
            The model was trained on a dataset on Kaggle and made as part of a project by Heicoder.
          </p>
        </div>
        <div class="col-7 rounded px-5">
          <div class="row pb-4">
            <h1>
              Customer Churn Prediction
            </h1>
            <p>
              Fill in the following details and hit submit to get the prediction!
            </p>
          </div>
        
          <div class="row">
            <div class="col-4">
              <label for="total_long_distance_fee" class="form-label">Total Long Distance Fee</label>
              <div class="input-group mb-3">
                <span class="input-group-text" id="basic-addon1">$</span>
                <input type="number" class="form-control" id="total_long_distance_fee" :placeholder="default_values.def_total_long_distance_fee" v-model="total_long_distance_fee" >
              </div>
            </div>
            <div class="col-4">
              <label for="total_charges_quarter" class="form-label">Total Charges Quarter</label>
              <div class="input-group mb-3">
                <span class="input-group-text" id="basic-addon1">$</span>
                <input type="number" class="form-control" id="total_charges_quarter" :placeholder="default_values.def_total_charges_quarter" v-model="total_charges_quarter" >
              </div>
            </div>
          </div>

          <div class="row">
            <div class="col-3 mb-3">
              <label for="tenure_months" class="form-label">Tenure Months</label>
              <input type="number" class="form-control" id="tenure_months" :placeholder="default_values.def_tenure_months" v-model="tenure_months">
            </div>
            <div class="col-3 mb-3">
              <label for="num_referrals" class="form-label">Num of Referrals</label>
              <input type="number" class="form-control" id="num_referrals" :placeholder="default_values.def_num_referrals" v-model="num_referrals">
            </div>
          </div>

          <div class="col-6 mb-3">
            <label for="num_dependents" class="form-label">Num of Dependents</label>
            <input type="number" class="form-control" id="num_dependents" :placeholder="default_values.def_num_dependents" v-model="num_dependents">
          </div>

          <div class="mb-3 col-3 form-check">
              <input type="checkbox" class="form-check-input"  v-model="married" id="married">
              <label class="form-check-label" for="married">Happily Married?</label>
            </div>
          <div class="col-6 pb-4">
            Contract Type
            <select class="form-select" v-model="contract_type">
              <option value="Month-to-Month" selected>Month to Month</option>
              <option value="One Year">One Year</option>
              <option value="Two Year">Two Year</option>
            </select>
          </div>
          <h5> Does the user have the following features?</h5>
            <div class="col-12 pb-4">
              <div class="form-check form-check-inline">
                <input class="form-check-input" type="checkbox" id="has_premium_tech_support" v-model="has_premium_tech_support">
                <label class="form-check-label" for="has_premium_tech_support" >Premium Tech Support</label>
              </div>
              <div class="form-check form-check-inline">
                <input class="form-check-input" type="checkbox" id="has_device_protection"  v-model="has_device_protection" >
                <label class="form-check-label" for="has_device_protection">Device Protection</label>
              </div>
              <div class="form-check form-check-inline">
                <input class="form-check-input" type="checkbox" id="has_online_backup"  v-model="has_online_backup">
                <label class="form-check-label" for="has_online_backup">Online Backup</label>
              </div>
            </div>
            <button type="submit" @click="submitForm()" class="btn btn-primary me-3">Get Prediction</button>
            <span class="text-danger fw-bold" v-if="prediction == 1">
              This customer is expected to churn!
            </span>
            <span class="text-success fw-bold" v-else-if="prediction == 0">
              This customer is expected to stay!
            </span>
        </div>
      </div>
    </div>
  </div>

</template>

<style scoped>

</style>

<script>
import axios from "axios";
export default {
  data(){
    return{
      default_values:{
        def_total_long_distance_fee: 500.00,
        def_total_charges_quarter: 125.00,
        def_tenure_months: 6,
        def_num_dependents: 0,
        def_married: "No",
        def_num_referrals: 0,
      },
      contract_type:"Month-to-Month",
      prediction: {
        "empty":"", 
        1: "This customer is expected to churn!",
        0: "This customer is expected to stay!",
      },
    }
  }, 
  methods: {
    submitForm() {
      console.log(process.env)
      console.log("http://" + process.env.VITE_VUE_APP_PREDICT_URL + "/predict")
      console.log(process.env)
      const formData = {
        total_long_distance_fee: this.total_long_distance_fee || this.default_values.def_total_long_distance_fee,
        total_charges_quarter: this.total_charges_quarter || this.default_values.def_total_charges_quarter,
        tenure_months: this.tenure_months || this.default_values.def_tenure_months,
        num_dependents: this.num_dependents || this.default_values.def_num_dependents,
        num_referrals: this.num_referrals || this.default_values.def_num_referrals,
        married: this.married || "No",
        contract_type: this.contract_type,
        has_premium_tech_support: this.has_premium_tech_support || "No",
        has_device_protection: this.has_device_protection || "No",
        has_online_backup: this.has_online_backup || "No",
      };
      axios.post(
        // "http://localhost:5000/predict",
        // process.env.VUE_APP_PREDICT_URL + "predict",
        process.env.VITE_VUE_APP_PREDICT_URL + "/predict",
        formData
      ).then((response) => {
        console.log("Response is", response.data);
        // Set response
        this.prediction = response.data.prediction;
        console.log(response.data.prediction)

      }).catch((error) => {
        console.log("Error is", error);
      })
    },
  },
};
</script>