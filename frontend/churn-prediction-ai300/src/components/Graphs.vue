<script setup>
import MetricsAcrossThresholds from './MetricsAcrossThresholds.vue';
import ROC from './ROC.vue';
</script>

<template>
    <div class="row bg-white ">
        <div class="col pt-3 text-center">
                <h3>
                    More information below!
                </h3>
        </div>
            <div class="row pt-5 d-flex justify-content-center text-center">
                <MetricsAcrossThresholds></MetricsAcrossThresholds>
            </div>
            <div class="row pt-5 d-flex justify-content-center text-center">
                <ROC></ROC>
            </div>
            <!-- <h3>
                Metrics across Thresholds
            </h3> -->

        <!-- <div class="row justify-content-center bg-primary">
            Metrics across Thresholds
        </div> -->
    </div>
</template>