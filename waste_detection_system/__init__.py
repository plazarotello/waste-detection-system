"""
Module for creating, training and testing models for paperboard and plastic 
waste detection. The tools and frameworks used in this implementation are:

- PyTorch
- torchvision
- torchmetrics
- lightning.ai (former PyTorch Lightning)


The entry point to the module is the :ref:`main script <main>`. Below there are
some examples of how to use the module:

.. code-block:: python
    :name: Loading and saving weights
    :caption: Loading and saving weights
    :linenos:

    path_to_a_model = base.MODELS_DIR / 'project-name' / 'model.ckpt'
    
    weights = main.load_weights_from_checkpoint(path_to_a_model,    # path or str
                                main.models.AVAILABLE_MODELS.SSD,   # type of model
                                2)                                  # number of classes

    main.save_weights(weights, base.MODELS_DIR / 'project-name' / 'model_weights.pt')

.. code-block:: python
    :name: Hyperparameter search
    :caption: Hyperparameter search
    :linenos:

    main.hyperparameter_search(
        name='project-name',                                # name of the project
        config='path_to_hyper_options.json',                # path to hyper-options.json
        dataset=train_dataset,                              # training dataset
        selected_model=main.models.AVAILABLE_MODELS.SSD,    # type of model
        num_classes=2,                                      # number of classes
        tll=1,                                              # Transfer Learning Level
        weights=weights,                                    # the weights we've loaded before
        find_batch_size=False,                              # turn on to find the maximum batch size allowed
        metric='training_loss'                              # 'training_loss', 'Validation_mAP'...
    )

.. code-block:: python
    :name: Training
    :caption: Training
    :linenos:

    main.train(
        train_dataset=train_dataset,                        # training dataset
        val_dataset=val_dataset,                            # validation dataset
        name='project-name',                                # project name
        config='path_to_config.json',                       # path to the configuration JSON
        num_classes=2,                                      # number of classes
        tll=1,                                              # Transfer Learning Level
        resortit_zw=0,                                      # 0 for ResortIT, 1 for ZeroWaste (for naming purposes)
        selected_model=main.models.AVAILABLE_MODELS.SSD,    # type of model
        limit_validation=False,                             # turn on if you want to validate only a subset of samples
        weights=weights,                                    # the seights we've loaded before (if you are not starting training from zero)
        metric='training_loss'                              # metric to optimize
    )

    main.train_hybrid(
        train_dataset=train_dataset,                        # training dataset
        val_dataset=val_dataset,                            # validation dataset
        name='project-name',                                # project name
        num_classes=2,                                      # number of classes
        selected_model=base.AVAILABLE_MODELS.SSD,           # type of model
        selected_classifier=base.AVAILABLE_CLASSIFIERS.SVC, # type of classifier
        weights=weights                                     # weights we've loaded before (the base model won't be training)
    )

.. code-block:: python
    :name: Testing
    :caption: Testing
    :linenos:

    main.test(
        checkpoint_path='path-to-best-checkpoint.ckpt',     # path to the checkpoint CKPT
        selected_model=main.models.AVAILABLE_MODELS.SSD,    # type of model
        resortit_zw=0,                                      # 0 for ResortIT, 1 for ZeroWaste (for naming purposes)
        test_dataset=test_dataset                           # test dataset
    )


Thanks to the contributors and community behind this frameworks; you made this
project posible.
"""

__version__ = '0.0.1'
