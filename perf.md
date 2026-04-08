<!--
 * @Author: zhouyuchong
 * @Date: 2026-04-08 13:05:49
 * @Description: 
 * @LastEditors: zhouyuchong
 * @LastEditTime: 2026-04-08 13:05:54
-->
=== Throughput vs Context Length ===

Context:    11 tokens (0.0K) | Throughput: 30.52 tok/s | Free: 2.47GB
Context:   125 tokens (0.1K) | Throughput: 24.36 tok/s | Free: 2.48GB
Context:   353 tokens (0.3K) | Throughput: 10.69 tok/s | Free: 2.48GB
Context:   581 tokens (0.6K) | Throughput: 10.70 tok/s | Free: 2.47GB
Context:  1153 tokens (1.1K) | Throughput: 10.25 tok/s | Free: 2.48GB
Context:  1728 tokens (1.7K) | Throughput: 9.36 tok/s | Free: 2.47GB

=== Summary ===
Recommend keeping context under ~16K tokens for stable performance on RTX 3060 12GB