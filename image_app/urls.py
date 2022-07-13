from django.urls import path, re_path
from .views import DetailProject, ListProjects, delete_image, DetailImages, get_mask, get_rois, get_superpixels, \
    get_superpixels_boundary, get_image, PlotEmbeding, get_db_data, make_patches, train_autoencoder, \
    add_roi_to_traintest, create_roi, get_model, get_prediction, retrain_model, get_prednext_image, \
    delete_project, add_project, make_embed, get_embed, get_roi, get_embed_csv

app_name = 'image_app'

urlpatterns = [
    path('', ListProjects.as_view(), name='list-projects'),
    path('proj/add', add_project, name='add-proj'),
    path('proj/<slug:proj_name>', DetailProject.as_view(), name='project'),
    path('proj/<slug:proj_name>/delete', delete_project, name='delete-proj'),
    path('proj/<slug:proj_name>/delete/<slug:image_id>', delete_image, name='delete-image'),


    path('proj/<slug:proj_name>/image/<slug:image_id>', DetailImages.as_view(), name='annotation'),
    path('proj/<slug:proj_name>/<slug:image_id>/image', get_image, name='get-image'),
    path('proj/<slug:proj_name>/<slug:image_id>/mask', get_mask, name='get-mask'),
    path('proj/<slug:proj_name>/<slug:image_id>/rois', get_rois, name='get-rois'),
    path('proj/<slug:proj_name>/<slug:image_id>/superpixel', get_superpixels, name='get-superpixels'),
    path('proj/<slug:proj_name>/<slug:image_id>/superpixelbound', get_superpixels_boundary, name='get-super-bound'),
    path('proj/<slug:proj_name>/model', get_model, name='get-model'),
    path('proj/<slug:proj_name>/<slug:image_id>/prediction', get_prediction, name='get-prediction'),
    path('proj/<slug:proj_name>/make_roi/<slug:image_id>', create_roi, name='create-roi'),
    path('proj/<slug:proj_name>/add_roi/<slug:roiid>/<slug:traintype>', add_roi_to_traintest, name='add-roi-to-traintest'),
    path('proj//<slug:proj_name>/image/<slug:image_id>/<slug:direction>', get_prednext_image, name='prev-next'),


    path('proj/<slug:proj_name>/get_embed/<slug:image_id>', get_embed, name='get-embed'),
    path('proj/<slug:proj_name>/get_roi/<slug:roi_id>', get_roi, name='get-roi'),
    path('proj/<slug:proj_name>/get_embed_csv', get_embed_csv, name='get-embed-csv'),
    path('proj/<slug:proj_name>/plot_embed', PlotEmbeding.as_view(), name='plot-embed'),


    path('proj/<slug:proj_name>/make_embed', make_embed, name='make-embed'),
    path('proj/<slug:proj_name>/make_patches', make_patches, name='make-patches'),
    path('proj/<slug:proj_name>/train_ae', train_autoencoder, name='train-ae'),
    path('prog/<slug:proj_name>/retrain_ae', retrain_model, name='retrain-ae'),

    re_path(r'^proj/db/[-%\w]+$', get_db_data, name='get-data'),
    # re_path(r'^proj/(?:project-(?P<proj_name>[-_\w]+)/add_roi/[.-_\w]+/train|test$', add_roi_to_traintest, name='add-roi-to-traintest'),
]
