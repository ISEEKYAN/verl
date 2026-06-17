Megatron Lite backend
=====================

Last updated: 06/17/2026.

Megatron Lite (``mlite``) is an experimental Megatron-family training backend
for verl. It keeps the backend glue outside the verl tree: the ``mlite``
checkout provides ``megatron.lite`` and the ``verl_mlite`` launcher/config
package used by the example scripts in this repository.

Install the backend
-------------------

Clone the active Megatron Lite checkout and install its verl integration:

.. code-block:: bash

   git clone https://github.com/ISEEKYAN/mlite
   pip install -e mlite/experimental/lite/examples/verl

Alternatively, keep the checkout outside the Python environment and set
``MLITE_ROOT`` when running a launcher. The scripts add both
``$MLITE_ROOT/experimental/lite`` and
``$MLITE_ROOT/experimental/lite/examples/verl`` to ``PYTHONPATH``.

Run an example
--------------

The DeepSeek-V4 examples use the ``mlite`` engine for training and vLLM for
rollout where applicable:

.. code-block:: bash

   MODEL_PATH=/path/to/deepseek-v4 \
   MLITE_ROOT=/path/to/mlite \
   OPTIMIZER=fsdp2 \
   bash examples/sft/gsm8k/run_deepseek_v4_megatron.sh

.. code-block:: bash

   MODEL_PATH=/path/to/deepseek-v4 \
   MLITE_ROOT=/path/to/mlite \
   OPTIMIZER=fsdp2 \
   bash examples/grpo_trainer/run_deepseek_v4_megatron.sh

``OPTIMIZER`` accepts ``dist_opt`` for the vanilla Megatron distributed
optimizer and ``fsdp2`` for the Megatron Lite FSDP2 wrapper. The DeepSeek-V4
launchers default to a 128-GPU mesh with PP4, EP8, CP4, full activation
recompute, and ``fsdp2``.

DeepSeek-V4 DSA note
--------------------

DeepSeek-V4 uses fused DSA kernels and is intended for the H100 GPU path. In
addition to the normal verl runtime, the critical DSA-only dependencies are
``nvidia-cutlass-dsl==4.5.2`` and a develop-branch
``nvidia-cudnn-frontend`` build that includes ``IndexerForwardSm90`` support.
The ``nvidia-cudnn-frontend`` 1.24.1 release does not provide the required SM90
DSA indexer.
