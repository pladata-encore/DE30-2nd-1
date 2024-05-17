package com.example.lottotest

import retrofit2.Call
import retrofit2.http.POST

interface ApiService {
    @POST("/predict")
    fun getData(): Call<ApiDataResponse>
}